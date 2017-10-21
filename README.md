# JapaneseCharacterRecognizerCNN

# Instructions on how to train the network and obtain the results

## Select the network to be used : M6_1 or M8
## Set this value in the model_fn parameter of the estimator

   mnist_classifier = learn.Estimator(
       model_fn=M8, model_dir=log_path + "/japcnn_Kanji_god_M8_try2")

## Select the script to be trained on :
Distinct label counts for script types


Script type   Distinct Classes
-----------   ----------------
HIRAGANA      75
KATAKANA      48
KANJI         881
ALL           1004

Change the corresponding class values in
onehot_labels = tf.one_hot(
      indices=tf.cast(labels, tf.int32), depth=881)
and..

logits = tf.layers.dense(inputs=dropout3, units=881, kernel_initializer=tf.contrib.layers.xavier_initializer(),
      bias_initializer=tf.zeros_initializer())

Directory structure expected:

Code

|----jp_recog_main.py

|----dataimport.py

dataset

|----ETL1

	|-----ETL1C_01
  
		# .
    
		# .
    
		# .
    
	|-----ETL1C_13
|----ETL8B

	|-----ETL8B2C1
  
		# .
    
		# .
    
		# .
    
	|-----ETL8B2C3


