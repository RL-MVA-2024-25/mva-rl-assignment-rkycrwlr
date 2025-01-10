#%%
from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
import numpy as np
from tqdm import tqdm
from sklearn.ensemble import RandomForestRegressor
import joblib
import random
import os
import torch
from evaluate import evaluate_HIV, evaluate_HIV_population
from multiprocessing import Pool
import copy
from dataclasses import dataclass

NJOBS = 45
qfunc_path = "src/Qfunc.pkl"

class ProjectAgent:
    def __init__(self):
        self.Q = None
        self.nb_actions = 4

    def greedy_action(self,Q,s,nb_actions):
        Qsa = []
        for a in range(nb_actions):
            sa = np.append(s,a).reshape(1, -1)
            Qsa.append(Q.predict(sa))
        return np.argmax(Qsa)
    
    def act(self, observation, use_random=False):
        a = self.greedy_action(self.Q,observation,self.nb_actions)
        return a

    def save(self, path):
        pass

    def load(self):
        self.Q = joblib.load(qfunc_path)

def seed_everything(seed: int = 42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed_all(seed)

def eval():
    seed_everything(seed=42)
    # Initialization of the agent. Replace DummyAgent with your custom agent implementation.
    agent = ProjectAgent()
    agent.load()
    # Keep the following lines to evaluate your agent unchanged.
    score_agent: float = evaluate_HIV(agent=agent, nb_episode=1)
    score_agent_dr: float = evaluate_HIV_population(agent=agent, nb_episode=15)
    print(f"\n{score_agent} | {score_agent_dr}")
    with open(file="score.txt", mode="a") as f:
        f.write(f"\n{score_agent} | {score_agent_dr}")
    return (score_agent > 2e10) and (score_agent_dr > 2e10)
#%%
env_rand = TimeLimit(
    env=HIVPatient(domain_randomization=True), max_episode_steps=200
)  # The time wrapper limits the number of steps in an episode at 200.
env = TimeLimit(
    env=HIVPatient(domain_randomization=False), max_episode_steps=200
)  # The time wrapper limits the number of steps in an episode at 200.
#%%

def greedy_action(Q,s,nb_actions=4):
    Qsa = []
    for a in range(nb_actions):
        sa = np.append(s,a).reshape(1, -1)
        Qsa.append(Q.predict(sa))
    return np.argmax(Qsa)

def collect_samples(argsCollect):
    s, _ = argsCollect.env.reset()
    S = []
    A = []
    R = []
    S2 = []
    D = []
    for _ in tqdm(range(argsCollect.horizon), disable=argsCollect.disable_tqdm):
        if argsCollect.Q is None:
            a = argsCollect.env.action_space.sample()
        else:
            if random.random() < argsCollect.rand_ratio:
                a = argsCollect.env.action_space.sample()
            else:
                a = greedy_action(argsCollect.Q,s)

        s2, r, done, trunc, _ = argsCollect.env.step(a)
        S.append(s)
        A.append(a)
        R.append(r)
        S2.append(s2)
        D.append(done)
        if done or trunc:
            s, _ = argsCollect.env.reset()
        else:
            s = s2
    S = np.array(S)
    A = np.array(A).reshape((-1,1))
    R = np.array(R)
    S2= np.array(S2)
    D = np.array(D)
    return S, A, R, S2, D

@dataclass
class ArgsCollect:
    env: 'HIVPatient'
    horizon: int
    Q: 'RandomForestRegressor'
    rand_ratio: float
    disable_tqdm: bool


def collect_samples_multiprocess(env, horizon, Q=None, rand_ratio=0.15):
    n_run = horizon // 200
    # f = lambda h: collect_samples(copy.deepcopy(env),h,Q,rand_ratio, disable_tqdm=True)
    with Pool(NJOBS) as pool:
        result = list(tqdm(pool.imap_unordered(collect_samples,[ArgsCollect(copy.deepcopy(env),200,Q,rand_ratio,True) for _ in range(n_run)]), total=n_run))
    S = np.concatenate([e[0] for e in result], axis=0)
    A = np.concatenate([e[1] for e in result], axis=0)
    R = np.concatenate([e[2] for e in result], axis=0)
    S2 = np.concatenate([e[3] for e in result], axis=0)
    D = np.concatenate([e[4] for e in result], axis=0)
    return S, A, R, S2, D

def rf_fqi(S, A, R, S2, D, iterations, nb_actions, gamma, disable_tqdm=False):
    nb_samples = S.shape[0]
    Q = None
    SA = np.append(S,A,axis=1)
    for iter in tqdm(range(iterations), disable=disable_tqdm):
        if iter==0:
            value=R.copy()
        else:
            Q2 = np.zeros((nb_samples,nb_actions))
            for a2 in range(nb_actions):
                A2 = a2*np.ones((S.shape[0],1))
                S2A2 = np.append(S2,A2,axis=1)
                Q2[:,a2] = Q.predict(S2A2)
            max_Q2 = np.max(Q2,axis=1)
            value = R + gamma*(1-D)*max_Q2
        Q = RandomForestRegressor(n_estimators=25, n_jobs=NJOBS, max_depth=15)
        shuffle_idx = np.arange(len(value))
        np.random.shuffle(shuffle_idx)
        Q.fit(SA[shuffle_idx],value[shuffle_idx])
    return Q

def generation_procedure(env_rand, env, nb_patients, iterations, fit_steps, memory_size, rand_ratio=.15, gamma=0.98):
    Q = None
    S,A,R,S2,D = np.zeros((0,6)),np.zeros((0,1)),np.zeros((0,)),np.zeros((0,6)),np.zeros((0,))
    horizon = nb_patients*200
    for iter in range(iterations):
        print(f"Iteration {iter+1}/{iterations}")
        if iter==0:
            new_S_r,new_A_r,new_R_r,new_S2_r,new_D_r = collect_samples_multiprocess(env_rand, horizon)
            new_S,new_A,new_R,new_S2,new_D = collect_samples_multiprocess(env, horizon)
        else:
            new_S_r,new_A_r,new_R_r,new_S2_r,new_D_r = collect_samples_multiprocess(env_rand, horizon, Q=Q, rand_ratio=rand_ratio)
            new_S,new_A,new_R,new_S2,new_D = collect_samples_multiprocess(env, horizon, Q=Q, rand_ratio=rand_ratio)
        S = np.concatenate([S,new_S_r,new_S],axis=0)[-memory_size:]
        A = np.concatenate([A,new_A_r,new_A],axis=0)[-memory_size:]
        R = np.concatenate([R,new_R_r,new_R],axis=0)[-memory_size:]
        S2 = np.concatenate([S2,new_S2_r,new_S2],axis=0)[-memory_size:]
        D = np.concatenate([D,new_D_r,new_D],axis=0)[-memory_size:]
        Q = rf_fqi(S, A, R, S2, D, fit_steps, 4, gamma)
        Q.set_params(n_jobs=1)
        joblib.dump(Q, qfunc_path)
        if eval():
            break
    return S, A, R, S2, D, Q

#%%

nb_patients=30
iterations=20
fit_steps=300
memory_size=100000

_,_,_,_,_,Q = generation_procedure(env_rand, env, nb_patients, iterations, fit_steps, memory_size)