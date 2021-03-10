import express from 'express';
import bodyParser from 'body-parser';
import cors from 'cors';

export default function (store) {

  const app = express();
  app.use(cors());
  app.use(bodyParser.json());


  // Add an event from Frankensteins AutoML
  app.post('/event', (req, res) => {
      const event = req.body;
      store.dispatch({ type: 'ADD_EVENT_ACTION', event });
      res.status(200).send();
  });

  app.listen(3000, () => {
    console.log('Listening on port 3000');
  });
}
