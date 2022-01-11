import torch
from django.shortcuts import render
from yolov5 import detect
from IPython.display import clear_output 
from .models1 import Img,visitor_detials
from django.contrib import messages
from .validators import validate_file_size
from django.conf import settings
import os
#from PIL import Image
#from .checktype import validate_file_type

def home(request):
    #from ipdb import set_trace; set_trace()
    #print("Inside Home")
    a = ""
    b = ""
    path = ""
    g= ""
    f= ""
    if request.method=="POST":
        try:
            save_ip(request)
            img1 = request.FILES['img']
            out_img = validate_file_size(img1)
            # out_img_type = validate_file_type(img1)
            if out_img:
                # img = Image.open(img1)
               # newImage = img.resize((1024, 768), Image.ANTIALIAS)
               #  imgk = Img(img= img1)
                # imgk.save()
                b = img1
                ab = ''.join(str(a).split('\\')[1:])
                new_path = str(ab) + "\\" + str(b)
                imgk = Img(img=img1, predited_img=new_path)
                imgk.save()
                clear_output()
                print(f"Setup complete. Using torch {torch.__version__} ({torch.cuda.get_device_properties(0).name if torch.cuda.is_available() else 'CPU'})")
                path = f"./media/image/{img1}"
                a=detect.run(weights  = './app/yolov5/runs/train/exp3/weights/best.pt', source = path)
                b = img1
                #print("sajid")
                f = len(list(os.walk('runs/detect')))-2
                abs_path=settings.MEDIA_ROOT2+'/detect/exp'+str(f)
                g = f"runs/detect/exp{f}"
                #print("_______________",g)
                return render(request, 'home.html', {'a': g, 'b': b, 'path': path})
            else:
                messages.error(request,"The maximum file size that can be uploaded is 5 MB or Unsupported file extension.")
                path = f"./media/default_image/actual.jpg"
                g = f"./media/default_image/"
                b = "predicted.jpg"
                return render(request, 'home.html', {'a': g, 'b': b, 'path': path})
        except Exception as e:
            print(e)
            messages.error(request,e)
    else:
        path= f"./media/default_image/actual.jpg"
        g= f"./media/default_image/"
        b="predicted.jpg"
        return render(request, 'home.html', {'a': g, 'b': b, 'path': path})


def get_visitor_ip(request):
    print(request.META)
    x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
    if x_forwarded_for is not None:
        ip = x_forwarded_for.split(',')[0]
    else:
        ip = request.META.get('REMOTE_ADDR')
    return ip

def save_ip(request):
    ip = get_visitor_ip(request)
    #print("The visitor ip.....", ip)
    visitor = visitor_detials(visitor_ip_addr=ip)
    result = visitor_detials.objects.filter(visitor_ip_addr=ip)
    if len(result) == 1:
        print("IP address already present")
    elif len(result) > 1:
        print("IP address present more than once")
    else:
        visitor.save()
        print("New IP visited")







