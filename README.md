# recently the algorithm has been moved to https://github.com/jjkke88/RL_toolbox

# trpo
trust region policy optimitztion base on gym and tensorflow

<p>There are three versions of trpo, one for decrete action space like mountaincar, one for decreate action space task with image as input like atari games, and the last for continuous action space for pendulems.</p>
<p>The environment is base on openAI gym.</p>
<p>part of code refer to rllab</p>

# dependency
<ul>
<li>tensorflow 0.10</li>
<li>prettytensor</li>
<li>latest openai gym</li>
</ul>

# constructure for code
<ul>
<li>baseline:baseline estimation of baseline function  <img src="http://www.forkosh.com/mathtex.cgi?V_\pi"> </li>
<li>checkpoint:folder to store model file, can not be delete or will cause some error</li>
<li>distribution:distribution base class, it can be used to calculate probability of distributions, for example Gaussian.</li>
<li>logger:have a Logger class for log data to .csv file</li>
<li>agent:for disperse action space and continous action space</li>
<li>log:store log file</li>
<li>experiment: contain many different main file, run main file can start trainning or testing</li>
<li>environment.py: environment</li>
<li>krylov.py: implement of some math method:conjugate gradient descent , calculating hessian matrix</li>
<li>parameters.py: config file</li>
<li>utils.py: implement of some basic function: getFlat, setFlat, lineaSearch</li>
</ul>

# recent work
<ul>
<li>imple multi-thread trpo run  python main_multi_thread.py to try</li>
<li>imple tensorflow distributed trpo</li>
<li>imple trpo multi-process</li>
</ul>

# future work
<ul>
<li>complete trpo with image as input</li>
</ul>


