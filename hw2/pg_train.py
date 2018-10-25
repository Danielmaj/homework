## Used the skeleton from Tambet here https://github.com/tambetm/homework/blob/master/hw2/pg_bare.py

import argparse
import os
import csv
import json
import time
import gym
import numpy as np

import tensorflow as tf
import keras
import keras.backend as K
from keras.layers import Lambda, Input, Dense
from keras import optimizers
from keras.models import Model


def get_arguments():

    parser = argparse.ArgumentParser()
    parser.add_argument('env_name', type=str)
    parser.add_argument('--exp_name', type=str, default='vpg')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--discount', type=float, default=1.0)
    parser.add_argument('--n_iter', '-n', type=int, default=100)
    parser.add_argument('--batch_size', '-b', type=int, default=1000)
    parser.add_argument('--max_timesteps', '-ep', type=float)
    parser.add_argument('--learning_rate', '-lr', type=float, default=5e-3)
    parser.add_argument('--reward_to_go', '-rtg', action='store_true')
    parser.add_argument('--dont_normalize_advantages', '-dna', action='store_true')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--n_experiments', '-e', type=int, default=1)
    parser.add_argument('--hidden_layers', '-l', type=int, default=2)
    parser.add_argument('--hidden_nodes', '-s', type=int, default=32)
    parser.add_argument('--act_function', type=str, default='tanh')
    parser.add_argument('--nn_baseline', '-bl', action='store_true')

    args = parser.parse_args()

    return args


# https://github.com/keras-team/keras/blob/master/examples/variational_autoencoder.py
# reparameterization trick
# instead of sampling from Q(z|X), sample eps = N(0,I)
# z = z_mean + sqrt(var)*eps
def sampling(args):
    """Reparameterization trick by sampling fr an isotropic unit Gaussian.
    # Arguments:
        args (tensor): mean and log of variance of Q(z|X)
    # Returns:
        z (tensor): sampled latent vector
    """
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(-0.5 * z_log_var) * epsilon


def create_model(observation_space, action_space,discrete, args):
    assert isinstance(observation_space, gym.spaces.Box)
    assert isinstance(action_space, gym.spaces.Box) \
        or isinstance(action_space, gym.spaces.Discrete)

    # TODO: Use given observation space, action space and command line parameters to create a model.
    # NB! Use args.hidden_layers and args.hidden_nodes for the number of hidden layers and nodes.
    # NB! Depending if action space is Discerete or Box you need different outputs and loss function.
    #     For Discrete you need to output probabilities of actions and use cross-entropy loss.
    #     For Box you need to output means of Gaussians and use mean squared error loss.

    # YOUR CODE HERE
    ob_dim = observation_space.shape[0]
    ac_dim = action_space.n if discrete else action_space.shape[0]

    inputs = Input(shape=(ob_dim, ))

    x = Dense(args.hidden_nodes, input_shape=(ob_dim, ), activation=args.act_function)(inputs)

    opt = optimizers.Adam(lr=args.learning_rate,clipvalue=10)

    for i in range(args.hidden_layers-1):
        x = Dense(args.hidden_nodes, activation=args.act_function)(x)

    if discrete:
        x = Dense(ac_dim, activation='softmax') (x)
        model = Model(inputs,[x])
        model.compile(optimizer=opt, loss='sparse_categorical_crossentropy')
    else:
        z_mean = Dense(ac_dim, name='z_mean')(x)

        #We want the variance to be a variable outside the model so we input a constant to the model
        k_constants = K.variable([[1]])
        fixed_input = Input(tensor=k_constants)
        z_log_var = Dense(ac_dim, name='z_log_var',activation='softplus',kernel_initializer='ones')(k_constants)
        # Custom loss function for keras
        def custom_loss(y_true, y_pred):
            return K.mean(K.square(z_mean-y_true)/2*(K.exp(-0.5*z_log_var)),axis=-1)

        # use reparameterization trick to push the sampling out as output
        z = Lambda(sampling, output_shape=(ac_dim,), name='z')([z_mean, z_log_var])
        model = Model([inputs,fixed_input],[z])

        model.compile(optimizer=opt, loss=custom_loss)

    model.summary()

    return model


def create_baseline(observation_space,args):

    ob_dim = observation_space.shape[0]

    inputs = Input(shape=(ob_dim, ))

    x = Dense(args.hidden_nodes, input_shape=(ob_dim, ), activation=args.act_function)(inputs)

    for i in range(args.hidden_layers-1):
        x = Dense(args.hidden_nodes, activation=args.act_function)(x)

    x = Dense(1, activation='linear') (x)

    model = Model([inputs],[x])
    opt = optimizers.Adam(lr=args.learning_rate,clipvalue=10)
    model.compile(optimizer=opt, loss='mean_squared_error')

    model.summary()

    return model


def train_model(model,observations, actions, advantages,args):
    # TODO: Use given observations, actions and advantages to train the model.

    # YOUR CODE HERE
    model.train_on_batch(observations,actions,sample_weight=advantages)


def train_baseline(baseline,observations,returns,args):
    # TODO: Use given observations and returns to train the baseline.

    # YOUR CODE HERE
    # We train baseline with mean zero std one.
    returns = np.concatenate(returns)
    r_mean = np.mean(returns)
    r_std = np.std(returns)

    norm_returns = (returns - r_mean) / (r_std + 1e-8)

    baseline.fit(observations,norm_returns,verbose=0)


def sample_trajectories(env,model,discrete,args):
    max_steps = args.max_timesteps #or env.spec.timestep_limit

    observations = []
    actions = []
    rewards = []
    total_steps = 0
    while total_steps < args.batch_size:
        episode_observations = []
        episode_actions = []
        episode_rewards = []

        obs = env.reset()
        done = False
        steps = 0
        while not done:
            # TODO: Use your model to predict action for given observation.
            # NB! You need to sample the action from probability distribution!
            if discrete:
                o = np.array([obs])
                prob = model.predict([o],batch_size=1)
                action = np.random.choice(np.arange(len(prob[0])), p=prob[0])
            else:
                o = np.array([obs])
                action = model.predict([o],batch_size=1)[0]

            episode_observations.append(obs)
            episode_actions.append(action)
            obs, reward, done, _ = env.step(action)
            episode_rewards.append(reward)

            steps += 1
            if args.render:
                env.render()
            if steps >= max_steps:
                break

        observations.append(episode_observations)
        actions.append(episode_actions)
        rewards.append(episode_rewards)
        total_steps += steps

    return observations, actions, rewards


def compute_returns(rewards, args):
    # TODO: Compute returns for each timestep.
    # NB! Use args.discount for discounting future rewards.
    # NB! Depending on args.reward_to_go calculate either total episode reward or future reward for each timestep.

    # YOUR CODE HERE

    returns = []

    if args.reward_to_go or args.nn_baseline :

        for ep_rewards in rewards:
            ep_returns = [0.] * len(ep_rewards)
            R = 0
            for i,r in enumerate(list(reversed(ep_rewards))):
                R = args.discount*R + r
                ep_returns[len(ep_rewards)-1-i] = R
            returns.append(ep_returns)
    else:

        for ep_rewards in rewards:
            R = 0
            for r in reversed(ep_rewards):
                    R = args.discount*R + r
            ep_returns = [R] * len(ep_rewards)
            returns.append(ep_returns)

    return returns





def compute_advantages(returns,observations,baseline,args):
    # TODO: Compute advantages as difference between returns and baseline.
    # NB! Depending on args.dont_normalize_advantages normalize advantages to 0 mean and 1 standard deviation.

    advantages = []
    advantages = np.concatenate(returns)

    if not args.dont_normalize_advantages:
        r_mean = np.mean(advantages)
        r_std = np.std(advantages)
        advantages = (advantages - r_mean) / (r_std + 1e-8)

    if args.nn_baseline:
            b_n = baseline.predict(observations,batch_size=len(observations))[:,0]
            if args.dont_normalize_advantages:
                r_mean = np.mean(advantages)
                r_std = np.std(advantages)
                b_n = r_mean + (r_std + 1e-8)*b_n
            advantages -= b_n

    #if not args.dont_normalize_advantages:
    #    advantages = returns.copy()
    #else:
    #    flat_ret = np.concatenate(returns)
    #    r_mean = np.mean(flat_ret)
    #    r_std = np.std(flat_ret)

    #    for ep_returns in returns:
    #        ep_advantages = [0.] * len(ep_returns)
    #        ep_advantages = (ep_returns - r_mean) / (r_std + 1e-6)
    #        advantages.append(ep_advantages)


    return advantages




def create_environment(args):

    print("Environment:", args.env_name)

    env = gym.make(args.env_name)

    print("Observations:", env.observation_space)
    print("Actions:", env.action_space)

    return env

def create_experiment_directory(exp,args):

    # create experiment directory
    logdir = os.path.join('data', args.exp_name + '_' + args.env_name, str(exp))
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    return logdir

def open_results_file(logdir):

    csvfile = open(os.path.join(logdir, 'log.txt'), 'w')
    csvwriter = csv.writer(csvfile, delimiter='\t')
    csvwriter.writerow(["Time", "Iteration", "AverageReturn", "StdReturn", "MaxReturn", "MinReturn",
                       "EpLenMean", "EpLenStd", "TimestepsThisBatch", "TimestepsSoFar"])
    return csvfile,csvwriter

def write_params_file(logdir,args):

    # write params file
    with open(os.path.join(logdir, 'params.json'), 'w') as f:
        json.dump(vars(args), f)

def main():

    args = get_arguments()

    env  = create_environment(args)

    discrete = isinstance(env.action_space, gym.spaces.Discrete)

    # loop over experiments
    for e in range(args.n_experiments):

        model = create_model(env.observation_space, env.action_space,discrete, args)

        baseline = create_baseline(env.observation_space,args)

        logdir = create_experiment_directory(e,args)

        csvfile,csvwriter = open_results_file(logdir)

        write_params_file(logdir,args)

        # main training loop
        total_timesteps = 0
        start = time.time()
        for i in range(args.n_iter):

            observations, actions, rewards = sample_trajectories(env, model,discrete,args)

            returns = compute_returns(rewards, args)

            # flatten observations and actions
            observations = np.concatenate(observations)
            actions = np.concatenate(actions)

            advantages = compute_advantages(returns,observations,baseline,args)

            # Train model and baseline
            train_baseline(baseline,observations,returns,args)
            train_model(model,observations,actions,advantages,args)

            # log statistics
            returns = [sum(eps_rew) for eps_rew in rewards]
            lengths = [len(eps_rew) for eps_rew in rewards]
            total_timesteps += len(observations)
            print("Iteration %d:" % (i + 1),
                  "reward mean %f±%f" % (np.mean(returns), np.std(returns)),
                  "episode length %f±%f" % (np.mean(lengths), np.std(lengths)),
                  "total timesteps", total_timesteps)
            csvwriter.writerow([time.time() - start, i,
                               np.mean(returns), np.std(returns),
                               np.max(returns), np.min(returns),
                               np.mean(lengths), np.std(lengths),
                               len(observations), total_timesteps])

        csvfile.close()
        # TODO: Optional - save the model for later testing.

    print("Done")


main()
