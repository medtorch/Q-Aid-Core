var Context = {
  Logo: require('../../assets/logo.png'),

  Onboarding: {
    SkipOnboarding: false,
  },

  Auth: {
    State: '',
  },

  Chat: {
    ChatImageSource: {
      uri:
        'https://icons-for-free.com/iconfiles/png/512/add+photo+plus+upload+icon-1320184027779532643.png',
    },
    ChatImageValue: null,

    ChatImagePending: null,
    ChatImageValuePending: null,
    ChatState: 'invalid',
  },
};
module.exports = Context;
