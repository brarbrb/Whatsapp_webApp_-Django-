We followed this tutorial to create a djano project: 
[Django Tutorial in Visual Studio Code](https://code.visualstudio.com/docs/python/tutorial-django#_go-to-definition-and-peek-definition-commands).


1. Create a project environment for the Django tutorial. We started with venv when working locally, and switched to conda on VM

Run in bash:
```bash
python3 -m venv .venv
source .venv/bin/activate # ensure selecting the python in the enviroment and not the global one
python -m pip install --upgrade pip
python -m pip install django
```

2. Create the Django project
```bash
django-admin startproject web_project . # this command assumes (by use of . at the end) that the current folder is your project folder
```
This creates different files and folders that are needed for django stored in `web_project` folder. 

```bash
python manage.py migrate # creates an empty development database
```
When you run the server the first time, it creates a default SQLite database in the file db.sqlite3 that is intended for development purposes, but can be used in production for low-volume web apps.

Start Django's development server using the command:
```bash
python manage.py runserver
```
If you want to use a different port than the default 8000, specify the port number on the command line, such as `python manage.py runserver 5000`.

3. Create a Django app
```bash
python manage.py startapp whatsapp
```