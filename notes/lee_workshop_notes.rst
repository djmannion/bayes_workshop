
Notes for workshop on Lee (2018)
================================

Introduction
------------

* Bayesian methods as using probability distributions to represent uncertainty, and how such distributions are updated by observing data.

* Approach taken here:

   * Not really going to consider Bayes' rule.
   * No 'coin flips' or arbitrary examples.
   * Follow the article in working through an example with psychophysics data.
   * Cover implementation issues by coding in Python as we go.

* **Aside**: What's a clue that the author used LaTeX? What did they do wrong?

* Different applications of Bayesian methods in psychology:

   * Bayes in the head. Examples? Lightness models, some predictive coding, Dan's work in general.
   * Stats. Replacing standard frequentist methods with Bayesian equivalents. Kruschke, Wagenmakers, JASP, etc.
   * Models.

      * Overly broad use of 'cognitive'?
      * Often can be similar to stats, but difference is that model ingredients and parameters are linked to psychological processes.


Advantages of Bayesian methods
------------------------------

* Principled. Underlying probability framework known and accepted. Less important, pragmatically.

* Allows creative freedom and flexibility. More important.

* Single mapping of parameters to data is an important practical point. No need for intervening summaries, goal is to 'explain' the individual observations.

* Can encompass different model structures:

   * Standard.
   * Hierarchical. Natural description of psychology experiments; e.g. participants within groups.
   * Latent mixture. Multiple sources could describe data.
   * Common cause. Linking different observations together. What structure does this also recapitulate?

* We will work through examples of these structures.

* Why do I like it? Difficult to explain at this point, without knowing more about what it is. Will come back to later.


Experimental data
-----------------

* On each trial, hear two beeps of differing duration. The first is always the standard (500ms). Task is to say which interval contained the longer beep.

* Also repeated but with visual flashes rather than beeps.

* **Question**: what's a problem with this design?

* There were 240 trials for each of the audio and visual conditions (3 blocks of 80 trials each), and 19 subjects. Not stated, but seems to use the method of constant stimuli (**Question**: what's MOCS?).

* **Activity**: Interpreting Figure 2.

* How to represent the data from an experiment in a consistent way? **Activity**: comparing 'tidy' and 'ndarray' methods.

* How many responses? 240 by 2 by 19 is 9120 responses.

* What other information do we need to know for each trial?

   * Which subject it is.
   * Which condition it is.
   * Which block it is.
   * Which trial it is.
   * What the comparison duration was.
   * What the subject's response was.

* Could encode this all *implicitly* as a ``19x2x3x80x2`` array, which would be 18240 numbers.

* Why not do it this way? Shapes different for each experiment, requires software to be aware of shape.

* 'Tidy' method is a more standard format, which always has two dimensions (a table). Now the information we need to know (as above) is stored in columns as explicit numbers, and each row is an observation. So we would need a ``9120x6`` array, which is 54720 numbers. We have tripled our data count.

* Setting up a project (``/code``, ``/data``, ``/results``). Usefulness in relation to codeocean.

* Emphasise template that can be adapted for other studies; may seem intimidating, but need to understand more than be able to recreate.

* Creating a basic ``conf.py``.

   * New Python3 features: ``types.SimpleNamespace``, ``pathlib``, ``print``.

* Using ``ipython`` to explore the Matlab file. Explain the use of ``pylab=qt``. Highlight the presence of ``NaN`` s in the data.

* Creating ``data.py``:

   * Writing ``get_columns()``.
   * Writing ``get_data()``:

      * Highlight: use of ``cols.index``
      * Difficulty: accessing attributes with a string.
      * Removal of ``NaN`` s. Different approaches?
      * Highlight: Data representation (float vs int)

* Data exploration:

   * Selecting a given participant.
   * Selecting a given participant and condition.
   * **Tip**: Always know the shape and data type of your variables.

* Visualisation:

   * Scatter one subject's psychometric function.
   * Emphasise that this is not for publication but for exploration.
   * Explain need to aggregate like in the figures.

* Creating ``utils.py`` and writing ``raw_to_prop``.

   * What are the inputs and outputs to such a function?
   * What is an algorithm that the function can use?

* Recreate Figure 2F; subject index is ``2``.


Research questions
------------------

* Analysis should be determined by the research question of interest.

* Key point is that the same framework is applicable to a variety of questions.


Model development
-----------------

* The logistic psychometric function as describing the probability of a particular response given a set of parameters and a stimulus value.

* We won't include ``s`` in our parameterisation.

* We won't worry about discussing the Cauchy. I don't think I have ever come across it in the psychophysics literature.

* **Question**: Isn't the author's interpretation of the alpha parameter wrong?

* Writing a logistic function in ``utils.py``.

* Reproduce Figure 3 in ipython using the function.

* In a Bayesian model, we need to put *priors* on the relevant parameters; here, the alpha (PSE) and beta (slope).

* Interesting point about the independence. What would dependence look like? Are there situations where they might be dependent?

* What sort of PSEs would we be expecting to 'see', given our theory? Also, remember the caveat about what the PSE represents here.

* What does the ``N(500, 50)`` distribution look like? Note that the distributions in this paper are described by their precision (inverse variance).

* For the slope, can establish constraints (soft and hard) on the lower and upper bounds and then how they are distributed within those bounds.

* Discuss the TruncatedGaussian distribution via ``scipy.stats.distributions.halfnorm``, and show examples with different standard deviations.

* We will look into checking the prior in a bit.


(Box) Models require a likelihood and a prior to make predictions
-----------------------------------------------------------------

* I get confused of what precisely is 'the likelihood'. I think of it as all the 'structural' parts of the model.

* Interesting point about the prior. You are really instantiating your theory, and it deserves higher consideration. Saying 'here is what I think the parameters are, given this theory'.


Graphical model representation
------------------------------

* I don't find these depictions to be all that intuitive, so we won't spend much time on them.

* However, the written description of the model beside it is of great interest.

* New aspect is the bottom, the Bernoulli; generates a 1 with probability ``p`` and a 0 with probability ``1 - p``.

* Key point is that this model (likelihood and priors) can *generate* data, of the same kind that are obtained in an experiment.


(Box) Graphical models have their limits
----------------------------------------

* Nothing particularly of relevance.


Prior prediction
----------------

* Given that we can use our model to generate data, 'prior prediction' is doing so in order to see that things are sensible.

* But before we can do that, we need to be able to write our models in Python. That is where ``pymc3`` comes in.

* Show the design of having a  ``models`` directory.

* Talk through the writing of ``demo_subj.py``, and the general format of having a ``get_model`` function that returns a ``model``. Note that the demo subject is index 6.

* Also explain why we have included the ``pf`` variable.

* Interactively, get the model and run ``sample_prior_predictive``. Reproduce the inset in Figure 5.

* Discuss the 'observations' as a key aspect of the prior 'predictives'.

* Talk through the writing of ``pred_to_dist`` in ``utils.py``. Note that this gets tricky.

* Reproduce Figure 6; ``scatter(a[:, 0], a[:, 1], s=a[:, 2], marker="s")``.

* How can we make this generalisable in code? Talk through the writing of ``prior_pred.py``.

   * Will need to cover dynamic importing using ``importlib``.
   * Will also need to cover taking in the model name as an option.
   * Will also need to cover random seeds.
   * Will also need to cover saving ``npz`` files.


Alternate models with vague priors
----------------------------------

* Distinction between 'informative', 'vague', 'flat' priors.

* Completely agree with Lee here.

* Create the ``demo_subj_vague.py`` model, using ``1000`` as the standard deviation priors. Reproduce Figures 8 and 9.

* Important point about parameterisation. Psychometric functions can use different formulations of the slope, and that matters for the prior.


(Box) Flat and uninformative priors are not the same thing
----------------------------------------------------------

* Definitely not the case that uniform (flat) is the same as uninformative.

* I'm still not clear on this though; to me, it is legitimately uninformative for the PSE, but 'informative' for the slope (in that it conveys strong predictions about the form of the function).


Parameter inference
-------------------

* Alright, now we have our model set up, we are ready to perform 'inference' by applying Bayes' rule.

* Basically, if we had the maths then we could write down an equation that would tell us the posterior probability distribution of our parameters (alpha and beta) given our likelihood and prior.

* But we usually don't - instead, we use a sampling procedure called Markov-Chain Monte-Carlo (MCMC) sampling. The idea is that this algorithm allows us to draw values (samples) from the joint posterior distribution. With enough samples, these draws approximate the posterior distributions that we would have worked out with maths.

* We won't go into the mechanics of how it works, mostly because I don't really know. But we will look into check that it does work OK.

* Run through an example trace procedure in ipython.

* Show the ``pymc3`` functions:

   * ``pm.plot_posterior``
   * ``pm.traceplot``
   * ``pm.autocorrplot``
   * ``pm.summary``
   * Also a scatter of the joint alpha and beta.

* Show the ``pf`` overlaid on the raw proportions to recreate Figure 13. Note that one point is off - I think that is because this subject had a ``NaN`` that the author didn't take into account.

* Walk through writing ``trace.py`` to make it more concrete, again.


(Box) Inference is not inversion
--------------------------------

* Important note that people call the sampling process 'fitting', and that the posterior characterises the 'best fit'. There isn't any real optimisation going on though - like I said before, if we knew the maths then we just evaluate the equation. The sampling aspect can make it seem like optimisation.

* Better to think of it as 'updating', from prior to posterior in light of data.

Posterior prediction
--------------------

* Does the model provide a reasonable account of the data? One way to tell is by looking at draws from the posterior. If those are systematically different from the actual data, something is awry.

* Run ``pm.sample_posterior_predictive`` in ipython. Reproduce Figure 14.

* Walk through writing ``post_pred.py``.


(Box) Describing data is not the same as predicting data
--------------------------------------------------------

* As a general rule, we need to be very careful when using the term 'predict'. In our usage, it is almost certainly going to be wrong. That is also the case with 'posterior predictives', as noted here.

* Interesting point about the priors actually being the predictions.

* Recapitulates the point about using posterior predictives; good isn't that informative, bad refutes.


Interpreting and summarising the posterior distribution
-------------------------------------------------------

* What do you do with the posterior? Say if you have done 50000 samples and have 10 parameters, then you have a 50000x10 set of numbers. Can be unwieldy.

* One strategy is marginalisation: summing over (integrating out) all of the other parameters to produce a one-dimensional vector of 50000 numbers.

* These can then be shown in a histogram, and summarised further by their mean and 95% highest posterior density interval ('credible interval').

* Show ``pm.plot_posterior`` again. Also show with ``kde_plot=True`` argument.


Model testing using prior and posterior distributions
-----------------------------------------------------

* How can we do hypothesis testing? Say if our hypothesis is that the subject is biased (their PSE is not equal to zero). We can define a null hypothesis that alpha is exactly zero and compare the evidence for that hypothesis against an alternative that the PSE is different from zero.

* Relative evidence can be evaluated by a 'Bayes factor'; this quantifies how many more times likely one model is than another in producing the observed data. There are rough guidelines to the sort of categories a given Bayes factor could be thought of (anecdotal, weak, strong, extreme), but can also just be interpreted continuously.

* If the null is a 'nested' version of the alternative (that is, the same except it is a particular instance of the alternative), can use the ratio of the prior and the posterior at the 'point of difference'. This is the Savage-Dickey ratio.

* I find the Savage-Dickey method pretty challenging philosophically. The alternative prior must have some mass at zero, in order for the calculation to work - but the alternative is meant to embody the hypothesis that the parameter *isn't* zero. I don't see how those things can be reconciled.

* However, we will see a more general strategy for computing Bayes factors. They end up the same as the Savage-Dickey method, in my experience - so it seems like just a handy shortcut rather than anything to think too deeply about.

* How can we compute it? We need to be able to evaluate the prior and the posterior at zero. That is straightforward for the prior; demonstrate use of ``scipy.stats.distributions.norm.pdf``.

* The posterior is tricker - all we have is a bunch of samples. How can we read off the density at a particular value? One reasonable approach is to use a kernel density estimate; talk though ``scipy.stats.gaussian_kde``.

* Also discuss about how it embodies the difficulty of finding evidence against small effects. Say if the prior was very tight, can we still obtain good evidence for the null?

* Evaluating the posterior can get tricky for beta, which is bounded. Not an issue here, but does come up. Need to use a KDE method that is aware of bounds; ``pyqt_fit`` is the best that I have found. Though this paper also mentions an interesting method where that use 'bins' of various sizes.


(Box) Model selection inferences based on parameter posteriors is perilous
--------------------------------------------------------------------------

* Relates to the question of hypothesis testing versus estimation.

* I agree with Lee and with Wagenmakers et al. (2018); need to first use model selection / hypothesis testing to work out if an effect 'exists' before worrying about estimating it's properties.

* Note that this goes against the Kruschke school of Bayes analysis.


Sensitivity analysis
--------------------

* If the specifics of a given prior are fairly arbitrary, then good to check how the particular arbitrary choice might have affected the posteriors.

* Important to note though that this only applies to priors with a degree of arbitrariness. If your priors are well specified by theory, then you don't want to have similar posteriors if you mess around with the prior.

* Can also change the likelihood, for example by including the potential for sequential effects that make successive trials non-independent.

* They propose an adjustment to ``p`` based on the previous decision.

* **Question:** This seems like a problem to me, as parameterised. Can get probabilities less than zero or greater than one?

* **Question:** Another subtlety regards what is being carried over. Is it the 'response' (i.e. pressing left or right) or the stimulus that response maps on to? They are confounded in this design, but won't be in most.


Latent-mixture modeling
-----------------------

* We have to deal with 'lapses', particularly when using naive first-year participants. This section describes a really interesting way of dealing with them.

* Basically, have two potential processes that can generate a response on a given trial. One is the stimulus-driven psychometric function, and the other is a stimulus-independent lapse function.

* Introduces new parameters to the demo subject model. Working 'backwards':

   * ``z`` is a binary variable with a value for every trial. It indicates whether the trial is a lapse (1) or a normal (0) trial.
   * ``phi`` is a single value that represents the probability that a given trial will be a lapse trial.
   * ``theta^c`` is a binary variable with what the lapse response 'would' be on every trial.
   * ``psi`` is a single value that represents the probability that a given lapse would be a 'longer' (1) or a 'shorter' (0) response.

* Talk through writing of the ``lapses.py`` model, and run it using ``trace.py``. Note that it will take about 10 minutes.

* Discuss the warning messages, and show the autocorrelation plots for the key variables.

* Look at the posterior of the ``phi`` parameter; this is akin to the 'lapse rate'.

* Plot the mean of the ``z`` parameter, over trials. This should reproduce the inset in Figure 21.

* **Question:** Is a U(0, 1) a reasonable prior for the lapse rate?

* **Question:** How should the response be conceptualised in the lapse trials - towards an interval or towards a stimulus?

* **Question::** How does this method behave with stimulus levels determined by an adaptive procedure? They don't typically have many out near the 'tails', so might be difficult to identify lapses?


(Box) Parameter estimation as model selection
---------------------------------------------

* Can use the same sort of 'indicator variable' approach to estimate a Bayes factor, rather than a Savage-Dickey.

* Note that these tend to be difficult to sample, in my experience. May need to investigate further using a ``pm.Mixture`` rather than a discrete variable.


Hierarchical modeling
---------------------

* Probably the most useful 'extra' concept.

* Can think of a continuum from treating different subjects as independent to identical. Hierarchical modeling allows a middle ground.

* Each subject's parameters are drawn from a parent distribution.

* Talk through coding ``group.py``. Need to mention the trickiness around the subject indices.

* Implement it without using ``testval`` first, and explain the error. Look at ``model.check_test_point()``.

* Set the ``testval`` arguments and re-run.

* Note the issues with the ``alpha_sd`` and ``beta_sd`` parameters. Set the upper limit of ``alpha_sd`` to 100 and re-run.

* Look at ``pm.forestplot(trace, varnames=["alphas"])``, and the same with betas.

* Discuss 'non-centred parameterisation', and change alphas

* Interesting to look at the paired trace of ``beta_mu`` and ``beta_sd``.


(Box) Cognitive and statistical models of individual differences
----------------------------------------------------------------

* The 'hierarchical' approach is similar to a random effects model, as is used in something like ANOVA.

* I don't quite follow the 'simple extension' using a latent mixture model.

* Interesting point about individual differences approaches lacking theory. Indeed, most predict that everyone would behave the same.


Finding invariances
-------------------

* Interesting points about 'invariances', and how such null results relate to the standard null-hypothesis frequentist testing framework.

* This is highly relevant to our work; are the psychometric parameters for two different parameters the same or different?

* Talk through the creation of ``av_cmp.py`` and run. Scatter the deltas against one another, should produce Figure 28B.

* Interesting point about the Krushke way of doing it; estimate separately and then perform inference on the samplewise difference. I don't really understand the justification though.


Common-cause modeling
---------------------

* Pretty straightforward; if you are willing to state that the two conditions are identical, then you can model them both with the same parameters. Because you have more data, the posterior estimates are more certain.

* Copy ``av_cmp.py`` to ``common_cause.py`` and convert it and run.

* But, can be biased if this assumption was wrong. If time, can compare the posterior predictives for the two conditions against the raw data.


(Box) The generality and paucity of common-cause models
-------------------------------------------------------

* Interesting comments about identify common causes.

* One example where we have tried to do something similar (kind of?) is in Elizabeth's experiment, where we had people do both scene and face recognition tasks to try and identify a common component.


Prediction and generalisation
-----------------------------

* Main point here is that it copes with 'missing' data well, because it can just be estimated.

* See for example Sol's experiment, where some subjects were good except for one condition where they had had enough. We can label that dodgy condition as 'missing' and estimate it from their performance on the other conditions and the performance of the other subjects.


Conclusion
----------

* Why a 'narrow but critical role'?

* What I like:

   * That it is all done in one step, and the raw data remains the fundamental unit.

   * Thinking in a generative way, about the sort of processes that could produce the observed data.

   * That the outcomes tend to be 'sensible' and readily interpretable.

   * That it is easy to tinker with the construction of a model.

* What I don't like:

   * It can be tricky to get the samplers to be well behaved.

   * It can be hard to pin down and defend an appropriate prior.

   * Hierarchical models, in particular, need lots of data and careful choice of model specification.

   * The surface level understanding of Bayesian methods seem easier than frequentist, but I'm worried that I'm missing the deeper aspects that flips that around.

   * ``pymc3`` can be tricky to interact with and debug, due to the use of the ``theano`` machine learning library.


