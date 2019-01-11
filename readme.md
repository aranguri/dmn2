# Tasks
Next steps
* train for a long time

If it doesn't work (in order)
* try to understand the code and check whether it's correct.
* making the code more concise
* dropout on word embeddings
* change eq 8 to a sofmtax

When it works
* allow different batch_sizes for training and test

# Logs
## 2019/8/1
Actual question: does the supervised gate system work?
It gets stuck around 0.48 without any further improvements. I don't understand why it isn't overfitting the training data.
Possible problem: the reshape in preprocess
The output from babi_parser has everything right: the sentences are the ones from the file, and the answer and the supporting facts indexes are correct.
Now, we are gonna check whether the values from next_batch are correct or not. First we are going to check the implementation and then the output data.

Both the implementation and the output seem to be correct.

Next, I'm going to debug model.py
It seems first_call doesn't have problems.
I feel there is a problem with indexing, but I'm not sure what's happening. Something interesting is that when it reaches its maximum performance (around .48) it always outputs a number greater or equal than the correct one.  Ohh, now I remember the error. It just outputs 1,3,5,7,9. It seems the DMN isn't correctly taking into account the input for the gates.

Other knowledge: it seems it isn't good for GPU processing because of lots of small operations. GPU seems to be good to large matrix multiplication.

In particular, I suspect from the memory_call function, and the swapping operations.
Also, another thing we could try is to make the problem easier. eg we can reduce the trainig set to 10 instead of 1000 trainig cases.


Lesson: don't: completely code and then test it. do: code and test simultanously. do: test after finishing each section.

From the data we have
first: blank, blank, ..., d1|2, d2|1
second: blank, blank, ..., d3|4, d4|3, d1|2, d2|1 (hence, it's not reading sequentially)

state: compare gates that are generated inside similarity fn and those outside

Current theory: the input is somewhat fixed. ie the nn has not that much freedom. if it wants to put 1 for a gate, it ends up being force to use 3 for the next one. how to test the hyphotesis. try saying that supporting facts are 1,3, 5 and vcheck 100% acc

It may not be able to make different values for the gates because of the input being too similar.

IT could be that the model doesn't have enough different values to then use them to make different values for the different gates. It's interesting that it can overfit to some cases, specificlally when question state has some specific input. I couldn't make the model overfit to decreasing order, yet.

The problem with gates was simple but subtle. tf.random.normal returns a new set of weights each time is executed. another reason to iteratively test the code as I'm writing it.



'''
think: is the focus on the last words (given by the rnn) good?
why are we using the same weight matrix in similarity measure?
google would be better i think if it redirects you to the first page and has a small square in the bottom right with the other results.
how do you say something that is valid in all the meta versions? For instance, I can say: be consistant in life. then, that rule applies to itself. So you need to be consistant in being consistant. But, if I limit myself to some scope (eg be consistant in programming) then I'm not talking about the consistantcy about following the rule (which is interesting, because not being consistant for the rule produces a similar effect to not being consistant for programming)
what if every neuron in a neural net is a GRUCell? It's prohibitely expensive, but who cares? How would be that structure?
This thing of talking to myself and listening to music to get more energies really help (I don't know if ¬sleeping is scalable though, but talkking to myself is) INdeed it made me more focused on what I was doing that days in the piecita/green where I was wondering in thoughts but having silence
wrt ww it seems I'm donig everythign wrong -- or worse, I'm doing nothing :)

Terms
Lazy evaluation: evaluating an expression iff it's needed
'''

With 1k cases, it reaches the answer 1,3 ,5, 7. However, with 10k it doesn't do that (yet)
It may be the case that the hyperparams are wrong.

possible explanation: weights are saturated
