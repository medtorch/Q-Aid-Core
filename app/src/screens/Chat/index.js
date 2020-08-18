import React, {useState, useCallback, useEffect} from 'react';
import {
  Icon,
  MenuItem,
  OverflowMenu,
  TopNavigation,
  TopNavigationAction,
  Avatar,
  Modal,
  Card,
  Button,
  Popover,
} from '@ui-kitten/components';
import {StyleSheet, View, Image} from 'react-native';
import {ChatStyle} from '../../components';
import PhotoUpload from 'react-native-photo-upload';

import {Auth} from 'aws-amplify';
import {GiftedChat} from 'react-native-gifted-chat';

import {get_reply} from './data.js';
import {MenuIcon, InfoIcon, ShareIcon, LogoutIcon, PhotoIcon} from './icons.js';
import {ChatContext} from './context.js';
import {User} from './user.js';

export function Main() {
  const [menuVisible, setMenuVisible] = React.useState(false);
  const [messages, setMessages] = useState([]);
  const [isTyping, setIsTyping] = useState(false);

  var replyIdx = 1;
  var ctx = new ChatContext();
  var user_ctx = new User();

  useEffect(() => {
    setMessages([generateReply(get_reply('intro'))]);
  }, []);

  const toggleMenu = () => {
    setMenuVisible(!menuVisible);
  };

  const renderMenuAction = () => (
    <TopNavigationAction icon={MenuIcon} onPress={toggleMenu} />
  );

  const renderOverflowMenuAction = () => (
    <React.Fragment>
      <OverflowMenu
        anchor={renderMenuAction}
        visible={menuVisible}
        onBackdropPress={toggleMenu}>
        <MenuItem accessoryLeft={InfoIcon} title="Models" />
        <MenuItem accessoryLeft={ShareIcon} title="Send to a doctor" />
        <MenuItem
          accessoryLeft={LogoutIcon}
          title="Logout"
          onPress={user_ctx.signOut}
        />
      </OverflowMenu>
    </React.Fragment>
  );

  const fetchPhotoCategory = (bs64img) => {
    var payload = {
      image_b64: bs64img,
    };

    fetch('https://q-and-aid.com/router', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(payload),
    })
      .then((response) => response.json())
      .then((responseData) => {
        console.log('got router response ', responseData);
        if (responseData['answer']) {
          ctx.category = responseData['answer'];
          setMessages((previousMessages) =>
            GiftedChat.append(
              previousMessages,
              generateReply('That looks like ' + responseData['answer']),
            ),
          );
        }
      })
      .catch((error) => {
        console.log('router error', error);
      });
  };
  const onPhotoUpload = async (file) => {
    if (file.error || typeof file.uri == 'undefined') {
      console.log('failed to load file ', file);
      return;
    }
    ctx.source.uri = file.uri;
  };
  const onPhotoSelect = (bs64img) => {
    ctx.reset();

    setMessages([generateReply(get_reply('intro'))]);

    var payload = {
      image_b64: bs64img,
    };

    fetch('https://q-and-aid.com/prefilter', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(payload),
    })
      .then((response) => response.json())
      .then((responseData) => {
        console.log('got prefilter response ', responseData);
        if (responseData['answer'] !== null && responseData['answer'] === 0) {
          ctx.state = 'valid';
          ctx.image_value = bs64img;
          fetchPhotoCategory(bs64img);
        }
      })
      .catch((error) => {
        console.log('prefilter error', error);
      });
  };

  const renderImagePicker = () => {
    const [modalVisible, setModalVisible] = React.useState(false);

    return (
      <View style={ChatStyle.container}>
        <Button
          appearance="ghost"
          status="basic"
          size="large"
          accessoryLeft={PhotoIcon}
          onPress={() => setModalVisible(true)}
        />

        <Modal
          visible={modalVisible}
          backdropStyle={ChatStyle.backdrop}
          onBackdropPress={() => setModalVisible(false)}>
          <Card disabled={true}>
            <PhotoUpload
              onResponse={onPhotoUpload}
              onPhotoSelect={onPhotoSelect}>
              <Image
                style={ChatStyle.modalImage}
                resizeMode="cover"
                source={ctx.source}
              />
            </PhotoUpload>
          </Card>
        </Modal>
      </View>
    );
  };

  const renderTitle = (props) => (
    <View style={ChatStyle.titleContainer}>
      <Avatar
        style={ChatStyle.logo}
        source={require('../../assets/logo.png')}
      />
    </View>
  );

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
    var msg = get_reply(cat);
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
    if (ctx.source !== 'valid') {
      console.log('asking q on invalid input');
      return cbk('error', 'invalid input');
    }

    console.log('asking q ', query);
    var payload = {
      image_b64: ctx.image_value,
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
      <TopNavigation
        alignment="center"
        accessoryLeft={renderImagePicker}
        title={renderTitle}
        accessoryRight={renderOverflowMenuAction}
      />
      <GiftedChat
        messages={messages}
        isTyping={isTyping}
        onSend={(messages) => onSend(messages)}
        user={user_ctx.user}
        renderUsernameOnMessage
        showUserAvatar={true}
      />
    </>
  );
}
