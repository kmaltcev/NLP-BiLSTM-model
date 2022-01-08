## User instructions

Here for the first time? Read along!

This platform is for experimenting with the impostors’ method in the field of authorship analysis and plagiarism
detection.<br>
The system implements the "Impostors Method" algorithm, which is highly known in the field of Natural Language
Processing.<br>
As a part of the algorithm, CNN, BiLSTM and Ensemble model using Soft Voting algorithm are utilized for learning and
classifying texts' to authors.<br>
Here, you can run your experiments and obtain detailed, visual results and performance reports.<br>

As you access the main page, you will be able to experiment for the name of science, please follow the next steps to
understand how to operate the system:

1. Open the sidebar if it is not already opened (Arrow at the top left of the screen)
2. Direct the system to the data by providing a full path to the data location. The data location is expected to be a
   folder, containing sub-folders, one for each author. The sub-folders should contain creations of the authors with
   `.txt` extensions.
3. Choose the required impostors by their names for the experiment, we allow multiple choices, i.e., when you choose 2
   authors for First Impostors and 2 authors for Second Impostors, they will be paired according to the order you chose
   them.
4. Choose the “Author under test” and “Creation under test” written by him. Please note that for the algorithm proper
   functioning, the questionable author should have more than one creation in his folder.
5. Before you run the experiment, on the bottom of the sidebar, you can see the Neural Network hyperparameters, which
   could be tuned to receive more accurate results.
6. Now you will be able to run “Analyze Authorship” and wait for the results.
7. After a while, you will see the results as bar-plots and our approximate prediction, makes you easier to analyze the
   authorship of “Creation under test”
