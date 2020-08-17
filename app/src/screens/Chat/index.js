import React, {useEffect} from 'react';
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
import {Chat} from './Chat.js';
import {Auth} from 'aws-amplify';
import {Context, ChatStyle} from '../../components';
import PhotoUpload from 'react-native-photo-upload';

const MenuIcon = (props) => <Icon {...props} name="more-vertical" />;

const InfoIcon = (props) => <Icon {...props} name="info" />;
const ShareIcon = (props) => <Icon {...props} name="share-outline" />;

const LogoutIcon = (props) => <Icon {...props} name="log-out" />;

const signOutAsync = async () => {
  try {
    await Auth.signOut();
  } catch (error) {
    console.log('error signing out: ', error);
  }
};

const PhotoIcon = (props) => <Icon {...props} name="image" />;

export function Main() {
  const [menuVisible, setMenuVisible] = React.useState(false);
  const [isLoaded, setIsLoaded] = React.useState(false);

  useEffect(() => {
    setTimeout(() => {
      setIsLoaded(true);
    }, 1000);
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
          onPress={signOutAsync}
        />
      </OverflowMenu>
    </React.Fragment>
  );

  const onPhotoUpload = async (file) => {
    if (file.error || typeof file.uri == 'undefined') {
      console.log('failed to load file ', file);
      return;
    }
    Context['Chat']['ChatImageSource'].uri = file.uri;
  };
  const onPhotoSelect = (bs64img) => {
    Context['Chat']['ChatImageValue'] = bs64img;
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
                source={Context['Chat']['ChatImageSource']}
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

  return (
    <>
      <TopNavigation
        alignment="center"
        accessoryLeft={renderImagePicker}
        title={renderTitle}
        accessoryRight={renderOverflowMenuAction}
      />
      <Chat />
    </>
  );
}
