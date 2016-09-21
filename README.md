# trpo
An implement of trust region policy optimization

<p>There are three versions of trpo, one for decrete action space like mountaincar, one for decreate action space task with image as input like atari games, and the last for continuous action space for pendulems.</p>
<p>The environment is base on openAI gym.</p>
<p>part of code refer to rllab</p>

# constructure for code
<ul>
<li>baseline:baseline estimation of baseline function  <img src="http://www.forkosh.com/mathtex.cgi?V_\pi">, note that baseline_tensorflow.py have some problems and can not be used now</li>
<li>checkpoint:folder to store model file, can not be delete or will cause some error</li>
<li>distribution:distribution base class, it can be used to calculate probability of distributions, for example Gaussian.</li>
<li>logger:have a Logger class for log data to .csv file</li>
<li>log:store log file</li>
<li>main.py: main file, run this file can start trainning or testing</li>
<li>agent.py: agent</li>
<li>environment.py: environment</li>
<li>krylov.py: implement of some math method:conjugate gradient descent , calculating hessian matrix</li>
<li>parameters.py: config file</li>
<li>utils.py: implement of some basic function: getFlat, setFlat, lineaSearch</li>
</ul>
