from django.shortcuts import render
import numpy as np
import cv2
# Create your views here.
from django.http import JsonResponse, HttpResponse  
from django.views import View
import json
import json

from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from django.core.files.uploadedfile import InMemoryUploadedFile
import os
import sys
from process.gesture_recognition_api import gesture_recognition

# gesture_recognition_api_path = os.path.abspath('/path/to/gesture_recognition_api/directory')
# sys.path.append(gesture_recognition_api_path)
MODEL_PATH = 'process/gesture_model.pt'
@method_decorator(csrf_exempt, name='dispatch')
class HelloWorldView(View):
    def get(self, request, *args, **kwargs):
        data = {"message": "Hello, World!"}
        return JsonResponse(data)

@method_decorator(csrf_exempt, name='dispatch')
class NameView(View):
    def post(self, request, *args, **kwargs):
        try:
            data = json.loads(request.body)  # parsing the json data from request body
            name = data.get('name', '')  # getting the 'name' field from data
            response_data = {"message": f"Hello, {name}!"}  # creating the response data
            return JsonResponse(response_data)  # returning the JsonResponse
        except json.JSONDecodeError:
            return HttpResponse("Invalid JSON data", status=400)
        
# @method_decorator(csrf_exempt, name='dispatch')
# class ImageSizeView(View):
#     def post(self, request, *args, **kwargs):
#         try:
#             image_file = request.FILES['image']  # getting the image file from the request
#             size = image_file.size  # getting the size of the image file
#             response_data = {"size": size}  # creating the response data
#             return JsonResponse(response_data)  # returning the JsonResponse
#         except KeyError:
#             return HttpResponse("No image file in request", status=400)
        
        
@method_decorator(csrf_exempt, name='dispatch')
class ImageSizeView(View):
    def post(self, request, *args, **kwargs):
        try:
            image_file = request.FILES['image']  # getting the image file from the request

            # convert image file to an numpy array
            npimg = np.fromstring(image_file.read(), np.uint8)

            # convert numpy array to image
            img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

            # convert image to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # calculate the size of grayscale image
            size = gray.size

            response_data = {"size": size}  # creating the response data
            return JsonResponse(response_data)  # returning the JsonResponse

        except KeyError:
            return HttpResponse("No image file in request", status=400)
        
@method_decorator(csrf_exempt, name='dispatch')
class ImagePredView(View):
    def post(self, request, *args, **kwargs):
        try:
            image_file = request.FILES['image']  # getting the image file from the request

            # convert image file to an numpy array
            npimg = np.fromstring(image_file.read(), np.uint8)

            # convert numpy array to image
            img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
            
            pred = gesture_recognition(img, MODEL_PATH)
            response_data = {"pred": pred}  # creating the response data
            return JsonResponse(response_data)  # returning the JsonResponse

        except KeyError:
            return HttpResponse("No image file in request", status=400)