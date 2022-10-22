from models import *
from dataset import *
from time import time

IS_TRAIN = True
IS_VAL = True
file_name = f'test2'

if __name__ == '__main__':
	train_data = get_data('../data/train.json')
	test_data = get_data('../data/val.json')

	# print(train_data[0].shape, train_data[1].shape)
	# print(test_data[0].shape, test_data[1].shape)
	# print(test_data[1])

	model = my_model((224,224,1),2)

	model.compile(optimizer=tf.optimizers.Adam(0.0001),
					loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
					metrics=[tf.metrics.sparse_categorical_accuracy],)


	checkpoint_filepath = os.path.join('cktp', f'{file_name}.h5')

	if IS_TRAIN:
		model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
			filepath=checkpoint_filepath,
			save_weights_only=True,
			monitor='val_sparse_categorical_accuracy',
			mode='max',
			save_best_only=True,
			verbose=1
			)

		history = model.fit(train_data[0], train_data[1],
			batch_size=32,
			epochs=300,
			validation_data=(test_data[0], test_data[1]),
			verbose=1,
			callbacks=[model_checkpoint_callback],
			)

		print('MAX ACC : ', max(history.history['val_sparse_categorical_accuracy']))

	if IS_VAL:
		model.load_weights(checkpoint_filepath)
		results = model.evaluate(test_data[0], test_data[1], batch_size=32)
		print("test loss, test acc:", results)