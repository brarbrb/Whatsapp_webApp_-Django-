# Working with node.js
Note when running node.js run it from the folder `wa-node`

(Run this command: `cd wa-node`)

When in folder run: 
```bash
npm init -y # creates a new Node.js project 
npm i axios express whatsapp-web.js qrcode # to install needed libraries 
```

1. `express` is used for exposing endpoints to connect the django to.
2. `axios` to send HTTP requests from Node to Django (sends state updates, unread messages)
3. `whatsapp-web.js` is the main reason we went with javascript. This is the package that allows to scrape, read and even reply to messages even though there's no official API
4. `qrcode` generates qr code that allows to connect to whatsapp as we used to with whatsapp web (connect through qr)


Run these lines to initialize the parameters for talking to django
```bash
export DJANGO_CALLBACK_URL=http://localhost:8000/ingest/session/ # set instead of export in windows cmd
export DJANGO_INGEST_BASE=http://localhost:8000/ingest/
export DJANGO_SECRET=newsecret 
export PORT=3001
```