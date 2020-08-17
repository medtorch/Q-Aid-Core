import React from 'react';
import {Auth, Storage} from 'aws-amplify';
import {GiftedChat} from 'react-native-gifted-chat';
import {Context} from '../../components';

import {useState, useCallback, useEffect} from 'react';

Storage.configure({level: 'private'});

const user = {
  _id: 1,
  name: 'Anonymous',
};

const replies = {
  intro: [
    "Let's start a new investigation! \n\nSend us a CT scan, a X-Ray or any other medical image, and let's discuss about it!",
  ],
  on_task: [
    "I'll analyze that asap!!",
    'Gimme a sec',
    'Working on it!',
    'On it',
    'This might take a while!',
  ],
  on_invalid_input: [
    'Stop sending me junk. Please ask a valid question!',
    "That's not a question!",
    "C'mon, I'm busy!",
    'Sorry, I cannot recognize that input. Try something else!',
  ],
  on_miss: [
    'Nothing found. Try something else!',
    'No verdict here!',
    'Sorry, no idea!',
    'Please try something else!',
  ],
};

var replyIdx = 1;

export function Chat() {
  const [messages, setMessages] = useState([]);
  const [isTyping, setIsTyping] = useState(false);

  useEffect(() => {
    Auth.currentSession()
      .then((data) => (user.name = data['accessToken']['payload']['username']))
      .catch((err) => console.log(err));

    setMessages([
      generateReply(
        replies['intro'][Math.floor(Math.random() * replies['intro'].length)],
      ),
    ]);
  }, []);

  const generateReply = (msg) => {
    replyIdx += 1;
    return {
      _id: replyIdx,
      text: msg,
      createdAt: new Date(),
      user: {
        _id: 2,
        name: 'Q&Aid',
        avatar:
          'https://cdn0.iconfinder.com/data/icons/avatar-2-3/450/23_avatar__woman_user-512.png',
      },
    };
  };

  const onReply = (cat, suff) => {
    var msg = replies[cat][Math.floor(Math.random() * replies[cat].length)];
    if (suff) msg += suff;

    setMessages((previousMessages) =>
      GiftedChat.append(previousMessages, generateReply(msg)),
    );
  };

  const isValidQuery = (input) => {
    if (input.length == 0) return 'invalid';
    if (input.trim().substr(-1) !== '?') return 'invalid';
    return 'valid';
  };

  const onQuestion = (query, cbk) => {
    var payload = {
      image_b64: Context['Chat']['ChatImageValue'],
      question: query,
    };

    fetch('https://q-and-aid.com/vqa', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(payload),
    })
      .then((response) => response.json())
      .then((responseData) => {
        console.log('got response ', responseData);
        if (responseData.answer) {
          return cbk('hit', responseData.answer);
        }

        return cbk('miss', 'no data');
      })
      .catch((error) => {
        return cbk('error', error);
      });
  };
  const onSend = useCallback((messages = []) => {
    if (messages.length == 0) return;

    setMessages((previousMessages) =>
      GiftedChat.append(previousMessages, messages),
    );
    var query = messages[0].text;
    var status = isValidQuery(query);

    if (status !== 'valid') {
      return onReply('on_invalid_input');
    }

    //onReply("on_task");
    setIsTyping(true);

    onQuestion(query, (status, data) => {
      console.log('VQA said ', status, data);
      setIsTyping(false);
      switch (status) {
        case 'hit': {
          return setMessages((previousMessages) =>
            GiftedChat.append(previousMessages, generateReply(data)),
          );
        }
        default: {
          return onReply('on_miss');
        }
      }
    });
  }, []);

  return (
    <>
      <GiftedChat
        messages={messages}
        isTyping={isTyping}
        onSend={(messages) => onSend(messages)}
        user={user}
        renderUsernameOnMessage
        showUserAvatar={true}
      />
    </>
  );
}
