# Plastic-UNet
The U-Net CNN with differentiable plasticity learning method implementation

## The learner architecture

The main learner part is based on U-Net architecture which demonstrate good prediction
power in medical images segmentation tasks. Taking into account that both tasks -
medical and seismical images segmentation - are very close we decided to use this
network architecture for this task. But, taking into account, that for seismical images
we need to identify specific parts of the images corresponding to the salt accumulations,
which rather is not clear segmentation task, we decided to augment the architecture with
synaptic plasticity rules. The introduction of plasticity rules allows to implement
long-term memory into our architecture allowing it to maximize influence of previously
seen training samples on the current optimization step.

## Plasticity rules

The synaptic plasticity is a major mechanism for continuous learning during life-time
implemented in the human brain, which makes it so efficient in assimilation of novel data
based on previous experience. The plasticity of the synaptic weights implemented by adjust
of weights during inference depending on training signals received from the environment.

In the ANN the plasticity rules can be implemented in different ways. In this work we will
consider plastic rules implemented separately from inference part of the ANN architecture.
We started with simple **Hebbian** rule, which stores plastic coefficients in Hebbian trace
during lifetime and applies learned plastic part at the final stage of inference routine, i.e.
before final layer of our network architecture. Then we continued with more advanced **Oja** rule
which provides workaround over Hebbian tendency to *decay plastic coefficients to zero* during
life-time. The Oja rule can maintain stable weight values in plastic part indefinitely in the
absence of stimulation, thus allowing stable long-term memories, while still preventing runaway divergences.

## Conclusion

The OJA rule gives considerable predictive performance boost over HEBB rule due to it's ability to maintain
learned weights indefinitely, thus allowing stable long-memories. Due to this effect knowledge inferred
 from previously seen training samples greatly determines the update in the current optimization step.
