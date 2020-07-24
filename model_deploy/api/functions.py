def handle_uploaded_file(f):
    with open('api/static/api/uploads/'+f.name, 'wb+') as destination:  
        for chunk in f.chunks():
            destination.write(chunk)
