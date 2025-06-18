from ultralytics import YOLO
import matplotlib.pyplot as plt



def train_yolo( ):
    """
    (Resumes) training the YOLOv11 model
    """
    model = YOLO("C:/Users/youri/OneDrive/BachelorThesis/classification/YOLOv11_classification/YOLOv11_classification_final_run/weights/epoch180.pt")
    model.train(resume=True, data="C:/Users/youri/BachelorThesisLocal/database/YOLO_training", device=0)


def finetune_yolo_edsr():
    """
    Fine-tunes the YOLOv11 model on EDSR denoised images
    """
    model = YOLO("C:/Users/youri/OneDrive/BachelorThesis/classification/YOLOv11_classification/YOLOv11_classification_final_run/weights/best.pt")
    data_dir = "C:/Users/youri/BachelorThesisLocal/database/YOLO_training_EDSR_finetuning_data"
    parameters = "C:/Users/youri/OneDrive/BachelorThesis/classification/YOLOv11_classification/YOLOv11_finetuned_edsr/args.yaml"
    results = model.train(data=data_dir,device =0, cfg=parameters )
    print(results)

def finetune_yolo_edm():
    """
    Fine-tunes the YOLOv11 model on EDM denoised images
    """
    model = YOLO("C:/Users/youri/OneDrive/BachelorThesis/classification/YOLOv11_classification/YOLOv11_classification_final_run/weights/best.pt")
    data_dir = "C:/Users/youri/BachelorThesisLocal/database/YOLO_training_EDM_finetuning_data"
    parameters = "C:/Users/youri/OneDrive/BachelorThesis/classification/YOLOv11_classification/YOLOv11_finetuned_edm/args.yaml"
    results = model.train(data=data_dir,device =0, cfg=parameters )
    print(results)


def evaluate_yolo_baseline(path_to_data):
    """
    Evaluates the baseline classifier performance
    """
    path_to_best_model = "C:/Users/youri/OneDrive/BachelorThesis/classification/YOLOv11_classification/YOLOv11_classification_final_run/weights/best.pt"
    model = YOLO(path_to_best_model)
    results = model.val(data=path_to_data, device=0)
    print(results)

def evaluate_yolo_edsr_upsampled_images():
    """
    Evaluates the EDSR Fine-tuned classifier
    """
    model = YOLO("C:/Users/youri/OneDrive/BachelorThesis/classification/YOLOv11_classification/YOLOv11_finetuned_edsr3/weights/best.pt")
    edsr_upsampled_data = "C:/Users/youri/BachelorThesisLocal/denoised/test1/edsr/upsampled_x1_classification_dir"
    results = model.val(data=edsr_upsampled_data, device=0, save_json=True, plots=True, save_txt=True, save_conf=True)
    print("Top-1 accuracy: ", results.top1)
    print("Top-5 accuracy: ", results.top5)
    print("Speed (ms per image): ", results.speed)

def evaluate_diff_denoise_steps_edm():
    """
    Evaluates the EDM Fine-tuned classifier on all denoised images from various numbers of denoising steps.
    """
    model = YOLO("C:/Users/youri/OneDrive/BachelorThesis/classification/YOLOv11_classification/YOLOv11_finetuned_edm3/weights/best.pt")
    diffusion_steps = [10,20,30,40,50,60,70,80,90,100]
    top_1,top_5,speed = [], [], []
    for i in diffusion_steps: #evaluates for all 10 different numbers of diffusion steps
        print(f"Classifying for {i} denoise steps.")
        edm_upsampled_data = f"C:/Users/youri/BachelorThesisLocal/denoised/test1/edm/edm/upsampled_out_{i}steps"
        results = model.val(data=edm_upsampled_data, device=0,save_json=True,plots=True,save_txt=True,save_conf=True)
        print(f"Top-1 accuracy at {i} denoising steps: ", results.top1)
        print(f"Top-5 accuracy at {i} denoising steps: ", results.top5)
        print(f"Speed (ms per image) at {i} denoising steps: ", results.speed, "\n")
        top_1.append(results.top1)
        top_5.append(results.top5)
        speed.append(results.speed.get('inference'))
    plot_steps_vs_accuracy(top_1,top_5,diffusion_steps)

def plot_steps_vs_accuracy(top1,top5,diffusion_steps):
    """
    Plots the accuracy scores accross the various denoising steps.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(diffusion_steps, top1, label='Top-1 Accuracy', marker='o')
    plt.plot(diffusion_steps, top5, label='Top-5 Accuracy', marker='s')
    plt.xlabel('Nr Diffusion Steps')
    plt.ylabel('Accuracy')
    plt.title('YOLO Classification Accuracy vs EDM Diffusion Steps')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("accuracy_vs_diffusion_steps.png")
    plt.show()

if __name__=="__main__":
    print("Evaluating the fine-tuned model on the EDM denoised images: ")
    evaluate_diff_denoise_steps_edm()
    print("Evaluating the fine-tuned model on the EDSR denoised images: ")
    evaluate_yolo_edsr_upsampled_images()
    print("Done!!")
