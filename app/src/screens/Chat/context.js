import React from 'react';

class ChatContext {
  constructor() {
    this.reset();
  }

  reset() {
    this.source = {
      uri:
        'https://icons-for-free.com/iconfiles/png/512/add+photo+plus+upload+icon-1320184027779532643.png',
    };
    this.image_value = null;

    this.state = 'invalid';
    this.category = null;
  }
}

module.exports.ChatContext = ChatContext;
