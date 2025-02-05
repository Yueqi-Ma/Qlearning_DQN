import numpy as np
import gym

# 创建 FrozenLake 环境
env = gym.make('FrozenLake-v1', render_mode='human')

# 初始化 Q 表
num_states = env.observation_space.n  # 状态数量
num_actions = env.action_space.n      # 动作数量
Q = np.zeros((num_states, num_actions))  # Q 表初始化为 0

# 超参数
learning_rate = 0.8   # 学习率
discount_factor = 0.95  # 折扣因子
num_episodes = 1000    # 训练的总回合数
max_steps = 100        # 每个回合的最大步数

# 探索参数
epsilon = 1.0          # 初始探索率
max_epsilon = 1.0      # 最大探索率
min_epsilon = 0.01     # 最小探索率
decay_rate = 0.01      # 探索率衰减率

# 训练 Q-Learning 算法
for episode in range(num_episodes):
    state = env.reset()[0]  # 重置环境，获取初始状态
    done = False            # 是否结束当前回合

    for step in range(max_steps):
        # 探索-利用权衡：选择动作
        if np.random.uniform(0, 1) > epsilon:
            action = np.argmax(Q[state, :])  # 利用：选择 Q 值最大的动作
        else:
            action = env.action_space.sample()  # 探索：随机选择动作

        # 执行动作，获取下一个状态和奖励
        new_state, reward, done, truncated, info = env.step(action)

        # 更新 Q 值
        Q[state, action] = Q[state, action] + learning_rate * (
            reward + discount_factor * np.max(Q[new_state, :]) - Q[state, action]
        )

        # 转移到下一个状态
        state = new_state

        # 如果回合结束，跳出循环
        if done:
            break

    # 衰减探索率
    epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)

    # 打印训练进度
    if episode % 100 == 0:
        print(f"Episode: {episode}, Epsilon: {epsilon:.2f}")

# 训练完成后测试智能体
state = env.reset()[0]
done = False
total_reward = 0

while not done:
    action = np.argmax(Q[state, :])  # 选择最优动作
    new_state, reward, done, truncated, info = env.step(action)
    total_reward += reward
    state = new_state
    env.render()  # 渲染环境

print(f"Total Reward: {total_reward}")

# 关闭环境
env.close()