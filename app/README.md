<img align="center" src="https://github.com/tudorcebere/Q-Aid/blob/master/misc/q_aid_logo_small1.png" alt="Q&Aid" width="75%">



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
