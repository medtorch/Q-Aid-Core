import React from 'react';
import {Auth} from 'aws-amplify';

class User {
  constructor() {
    this.reset();
    this.load();
  }

  reset() {
    this.user = {
      _id: 1,
      name: 'Anonymous',
    };
  }
  load() {
    var self = this;
    Auth.currentSession()
      .then(
        (data) => (self.user.name = data['accessToken']['payload']['username']),
      )
      .catch((err) => console.log(err));
  }

  async signOut() {
    try {
      await Auth.signOut();
    } catch (error) {
      console.log('error signing out: ', error);
    }
  }
}
module.exports.User = User;
