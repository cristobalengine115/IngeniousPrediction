from django.shortcuts import render

# Create your views here.
def index(request):
    return render(request, 'index.html')

def GUI(request):
    return render(request, 'guiMineria.html')