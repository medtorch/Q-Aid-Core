import Amplify from '@aws-amplify/core';
import awsconfig from '../aws-exports';

import {Authentication, Home} from './screens';

Amplify.configure({
  ...awsconfig,
  Analytics: {
    disabled: true,
  },
});

const App = Authentication(Home);

export default App;
