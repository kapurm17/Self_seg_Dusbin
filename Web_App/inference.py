from predict import recreate_model, get_tensor

idx_to_cls = {0:'cardboard' ,1: 'glass', 2: 'metal' , 3: 'paper',4: 'plastic', 5:'trash'}

model = recreate_model('weights.pth')

def get_class(img_bytes):
    tensor= get_tensor(img_bytes)
    output = model(tensor)
    index = output.data.cpu().numpy().argmax()
    return idx_to_cls[index]

