# Q&AID app

![React Native logo](https://reactnative.dev/img/header_logo.svg | width=40)
![PyTorch Logo](https://github.com/pytorch/pytorch/blob/master/docs/source/_static/img/pytorch-logo-dark.png)
<img src="https://reactnative.dev/img/header_logo.svg" width="20%" >
<img src="https://github.com/pytorch/pytorch/blob/master/docs/source/_static/img/pytorch-logo-dark.png" width="60%" >


## Run server

```
rm -rf node_modules
yarn install
rm -rf /tmp/metro-*

watchman watch-del-all
yarn start --reset-cache
```

## Run Android

```
react-native run-android
```
