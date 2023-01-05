import tensorflow as tf
import PIL


model = tf.keras.models.load_model("best_model_augmented.h5")


img = tf.keras.utils.load_img(
    "/home/marko/Desktop/doggo.jpg", target_size=(200, 200))
img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create a batch

predict = model.predict(img_array)

if predict[0] > 0.5:
    print("It's a dog.")
else:
    print("It's a cat.")

print(tf.config.list_physical_devices('GPU'))