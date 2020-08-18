module.exports.templates = {
  replies: {
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
    on_upload: ['That looks like a {{}}!'],
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
  },
};

module.exports.get_reply = function (type) {
  return module.exports.templates.replies[type][
    Math.floor(Math.random() * module.exports.templates.replies[type].length)
  ];
};
