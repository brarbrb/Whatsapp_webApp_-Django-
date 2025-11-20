const { Client, LocalAuth } = require('whatsapp-web.js');
const qrcode = require('qrcode-terminal');

const client = new Client({
    authStrategy: new LocalAuth()
});

client.on('qr', (qr) => {
    qrcode.generate(qr, { small: true });
});

client.on('ready', () => {
    console.log('Client is ready!');
    
    // Scrape last received message
    // client.getChats().then(chats => {
    //     chats.forEach(chat => {
    //         chat.getMessages({ limit: 1 }).then(messages => {
    //             if (messages.length > 0) {
    //                 const lastMsg = messages[0];
    //                 console.log(`Chat: ${chat.name}`);
    //                 console.log(`Last Message: ${lastMsg.body}`);
    //                 console.log(`From: ${lastMsg.from}`);
    //                 console.log(`Timestamp: ${new Date(lastMsg.timestamp * 1000)}`);
    //                 console.log('---');
    //             }
    //         });
    //     });
    // });
    
}); 

// on receiving new messages
client.on('message_create', async (message) => {
    const contact_object = await message.getContact();
    console.log(`Sender: ${contact_object.name}`);
    console.log(`Last Message: ${message.body}`);
    console.log(`Timestamp: ${new Date(message.timestamp * 1000)}`);
    console.log('---');
    // console.log('Full message object:', JSON.stringify(message, null, 2));
});

client.initialize();
