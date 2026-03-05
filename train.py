# Training loop
from agent.ppo_agent import PPOAgent
from env.snake_env import SnakeEnv
from tools.beautyful_progress_bar import PBar
import os, json, torch, time, argparse, numpy as np
from torch import nn


TRAINING_TIMESTEPS = 2_000_000  # Number of timesteps the model will be trained for
CHECKPOINT_INTERVAL = 500_000  # Number of steps between checkpoint saves
VISUALIZE_FREQUENCY = 500_000  # Number of steps after wich the agent will be playing a live game, to see how it's doing


if __name__ == "__main__":
    # Make the dirs to save training data
    os.makedirs(os.path.join('results', 'config'), exist_ok=True)
    os.makedirs(os.path.join('results', 'logs'), exist_ok=True)
    os.makedirs(os.path.join('results', 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join('results', 'model'), exist_ok=True)


    # Handle params
    parser = argparse.ArgumentParser(description='trainig arguments')
    
    parser.add_argument('--train-ts', type=int, default=None, help='Number of timesteps the model will be trained for')
    parser.add_argument('--ci', type=int, default=None, help='Number of steps between checkpoint saves')
    parser.add_argument('--vf', type=int, default=None, help='Number of steps after wich the agent will be playing a live game, to see how it\'s doing')
    args = parser.parse_args()
    
    if args.train_ts is not None:
        TRAINING_TIMESTEPS = args.train_ts
    if args.ci is not None:
        CHECKPOINT_INTERVAL = args.ci
    if args.vf is not None:
        VISUALIZE_FREQUENCY = args.vf

    
    # Initialize agent and env
    # env = SnakeEnv(gridW=10, gridH=10)
    env = SnakeEnv(gridW=6, gridH=6)
    agent = PPOAgent(env)


    # Save training configurations
    config = {
        'agent': {
            'timesteps_per_batch': agent.timesteps_per_batch,             # timesteps per batch (a batch is a number of timesteps before updating PPO's policy)
            'max_timesteps_per_episode': agent.max_timesteps_per_episode, # timesteps per episode (an episode is a game inside the env)
            'gamma': agent.gamma,
            'n_updates_per_iteration': agent.n_updates_per_iteration,       # Number of epoch, used to perform multiple updates on the actor and critic networks
            'clip': agent.clip,
            'lr': agent.lr,
        },
        'env': {
            'max_steps': env.max_steps,
            'obs_shape': env.observation_space.shape,
            'action_shape': env.action_space.shape,
        },
    }

    with open(os.path.join('results', 'config', 'training_config.json'), 'w') as f:
        json.dump(config, f, indent=4)


    # Initialize pointer to file where the stats for each batch will be saved
    training_logs_file = open(os.path.join('results', 'logs', 'training_logs.json'), 'w')
    training_logs_file.write('[\n')
    training_logs_first_entry = True  # Used to manage commas inside the json



    # =============== TRAINING LOOP (Copied from PPOAgent learn method) ===============
    
    t_so_far = 0 # Timesteps simulated so far
    batch_n = 0  # Batch number
    pbar = PBar(TRAINING_TIMESTEPS, preset="training")
    
    while(t_so_far < TRAINING_TIMESTEPS):
        batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_rews, batch_lens = agent.rollout()
        batch_n += 1

        # Calculate how many timesteps we collected this batch
        total_batch_steps = np.sum(batch_lens)
        t_so_far += total_batch_steps
        pbar.update(total_batch_steps)

        # Display batch results
        agent._print_stats(total_batch_steps, batch_rews, batch_lens, batch_n)


        # Save batch data in log file to make graphs after training
        avg_reward = float(sum(sum(ep_rews) for ep_rews in batch_rews) / len(batch_rews))
        max_reward = float(max(sum(ep_rews) for ep_rews in batch_rews))
        min_reward = float(min(sum(ep_rews) for ep_rews in batch_rews))
        avg_ep_len = float(sum(batch_lens) / len(batch_lens))

        log_data = {
            'batch': batch_n,
            'timesteps_so_far': int(t_so_far),
            'batch_steps': int(total_batch_steps),
            'avg_reward': avg_reward,
            'max_reward': max_reward,
            'min_reward': min_reward,
            'avg_episode_length': avg_ep_len,
            'max_episode_length': int(max(batch_lens)),
            'min_episode_length': int(min(batch_lens)),
        }

        if not training_logs_first_entry:
            training_logs_file.write(',\n')

        json.dump(log_data, training_logs_file, indent=4)
        training_logs_file.flush()  # To write immediately on disk
        training_logs_first_entry = False


        # Calculate V_{phi, k}
        V, _ = agent.evaluate(batch_obs, batch_acts)

        # PPO ALG STEP 5 (step 4 is in rollout function)
        # Calculate advantage
        A_k = batch_rtgs - V.detach()

        # Normalize advantages
        A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)

        for _ in range(agent.n_updates_per_iteration):
            # Epoch code (where we perform multiple updates on the actor and critic networks)

            # Calculate pi_theta(a_t | s_t)
            V, curr_log_probs = agent.evaluate(batch_obs, batch_acts)
            
            # Calculate ratios
            ratios = torch.exp(curr_log_probs - batch_log_probs)

            # Calculate surrogate losses
            surr1 = ratios * A_k
            surr2 = torch.clamp(ratios, 1 - agent.clip, 1 + agent.clip) * A_k

            actor_loss = (-torch.min(surr1, surr2)).mean()
            critic_loss = nn.MSELoss()(V, batch_rtgs)
            
            # Calculate gradients and perform backward propagation for actor network
            agent.actor_optim.zero_grad()
            actor_loss.backward(retain_graph=True)
            agent.actor_optim.step()

            # Calculate gradients and perform backward propagation for critic network
            agent.critic_optim.zero_grad()
            critic_loss.backward()
            agent.critic_optim.step()

        # =============== SAVE CHECKPOINT ===============
        if t_so_far % CHECKPOINT_INTERVAL < total_batch_steps and t_so_far != TRAINING_TIMESTEPS:
            clean_checkpoint_number = int(t_so_far / CHECKPOINT_INTERVAL) * CHECKPOINT_INTERVAL
            agent.save(total_timesteps=t_so_far, path=os.path.join('results', 'checkpoints'), file_name=f'checkpoint_{clean_checkpoint_number:,}.pth')
        
        # =============== VISUALIZE GAME ===============
        # N.B.: This visualize also the first batch to show the starting point (to avoid it delete 'or t_so_far == total_batch_steps')
        if t_so_far % VISUALIZE_FREQUENCY < total_batch_steps or t_so_far == total_batch_steps:
            stop = False
            tot_reward = 0
            obs, info = env.reset()

            while not stop:
                # Chose an action
                action, _ = agent.get_action(obs, deterministic=True)

                # Perform action
                obs, reward, terminated, truncated, info = env.step(action)
                stop = terminated or truncated

                tot_reward += reward

                # Visualize changes
                env.render()
                time.sleep(0.2)

            # Visualize results
            score = info['score']
            steps = info['steps']
            
            print('Game over:')
            print(f'Score: {score}')
            print(f'Reward: {tot_reward:.2f}')
            print(f'Steps performed: {steps}')
            print()

    pbar.close()
    
    # Close log file
    training_logs_file.write('\n]')
    training_logs_file.close()
            

    # Save final model
    print('Saving final model...')

    agent.save(total_timesteps=t_so_far, file_name='final_model.pth')

    print('Final model saved!')
    print('Training completed')