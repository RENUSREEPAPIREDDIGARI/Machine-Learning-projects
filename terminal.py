import tensorflow as tf 
import keras 
import numpy as np
import tkinter as tk 
from tkinter import filedialog
from tkinter.ttk import Label,Button
model=tf.keras.models.load_model("model")
def brain_mri_scan_img_selection():
    img_path=filedialog.askopenfilename()
    test_image = tf.keras.utils.load_img(img_path,target_size=(176,176))
    test_image = tf.keras.utils.img_to_array(test_image)
    test_image = np.expand_dims(test_image,axis=0)
    impred = model.predict(test_image)

    def roundoff(arr):
        # To round off according to the argmax of each predicted label array.

        arr[np.argwhere(arr != arr.max())] = 0
        arr[np.argwhere(arr == arr.max())] = 1
        return arr

    for classpreds in impred:
        impred = roundoff(classpreds)
    
    classcount = 1
    for count in range(4):
        if impred[count] == 1.0:
            break
        else:
            classcount+=1
    
    classdict = {1: "Mild Dementia", 2: "Moderate Dementia", 3: "No Dementia, Patient is Safe", 4: "Very Mild Dementia"}
    print(classdict[classcount])

root=tk.Tk()
root.title("Alzheimers Diagnosis Predictive Model using CNN Deep Learning  Algorithm")
root.geometry("500x100")
label=Label(master=root,text="Alzheimers Diagnosis Predictive Model using CNN Deep Learning  Algorithm")
label.pack()
button=Button(master=root,text="Select Brain MRI Scan Image",command=brain_mri_scan_img_selection)
button.pack()
root.mainloop()
