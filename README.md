# Snake-RL

<table align="center">
  <tr>
    <td>
      <img src="https://upload.wikimedia.org/wikipedia/commons/5/55/Snake_can_be_completed.gif" alt="Snake Game GIF" width="320"/>
    </td>
  </tr>
</table>

Reinforcement Learning (RL) algorithms have
attracted significant attention from the scientific
community due to their ability to learn optimal
behaviors through direct interaction with the environment. 
This project focuses on the design and
implementation of RL algorithms capable of training agents 
to play the game Snake. Algorithms
such as Advantage Actor-Critic (A2C), Deep Q-
Networks (DQN), and Double Deep Q-Networks
(DDQN) were employed. The report outlines the
strengths and limitations of each RL approach in
the context of the Snake game, along with the
training outcomes.

### Evaluation

To replicate the exact same environment use the command

```bash
conda env create -f environment.yml
conda activate snake
```

Make sure to have [Anaconda](https://www.anaconda.com/download) or [Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/main) before the installation.

I've also uploaded the ```requirements.txt``` file if you want to create your own virtual environment.
Use this command to install everything inside your venv.

```bash
pip install -r requirements.txt
```

After these steps, to run the evaluation your have to simply run:

```bash
python evaluate.py
```