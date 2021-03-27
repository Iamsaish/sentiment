import React, { useEffect, useState } from 'react';
import './App.css';
import {loadLayersModel, tensor2d, ready} from '@tensorflow/tfjs';
import padSequences from './helper/paddedSeq'
import   {
  TextField,
  Grid,
  AppBar,
  Toolbar,
  Typography,
  IconButton,
  Button,
  


} from '@material-ui/core'
import { makeStyles } from '@material-ui/core/styles';


function App() {
  const useStyles = makeStyles((theme) => ({
    root: {
      flexGrow: 1,
    },
    menuButton: {
      marginRight: theme.spacing(2),
    },
    title: {
      flexGrow: 1,
    },
  }));
  const classes = useStyles();

  const url = {

    model: 'https://storage.googleapis.com/tfjs-models/tfjs/sentiment_cnn_v1/model.json',
    metadata: 'https://storage.googleapis.com/tfjs-models/tfjs/sentiment_cnn_v1/metadata.json'
};

const OOV_INDEX = 2;

const [metadata, setMetadata] = useState();
const [model, setModel] = useState();
const [testText, setText] = useState("");
const [testScore, setScore] = useState("");
const [trimedText, setTrim] = useState("")
const [seqText, setSeq] = useState("")
const [padText, setPad] = useState("")
const [inputText, setInput] = useState("")


async function loadModel(url) {
  try {
    const model = await loadLayersModel(url.model);
    setModel(model);
  } catch (err) {
    console.log(err);
  }
}

async function loadMetadata(url) {
  try {
    const metadataJson = await fetch(url.metadata);
    const metadata = await metadataJson.json();
    setMetadata(metadata);
  } catch (err) {
    console.log(err);
  }
}


const getSentimentScore =(text) => {
  console.log(text)
  const inputText = text.trim().toLowerCase().replace(/(\.|\,|\!)/g, '').split(' ');
  setTrim(inputText)
  console.log(inputText)
  const sequence = inputText.map(word => {
    let wordIndex = metadata.word_index[word] + metadata.index_from;
    if (wordIndex > metadata.vocabulary_size) {
      wordIndex = OOV_INDEX;
    }
    return wordIndex;
  });
  setSeq(sequence)
  console.log(sequence)
  // Perform truncation and padding.
  const paddedSequence = padSequences([sequence], metadata.max_len);
  console.log(metadata.max_len)
  setPad(paddedSequence)

  const input = tensor2d(paddedSequence, [1, metadata.max_len]);
  console.log(input)
  setInput(input)
  const predictOut = model.predict(input);
  const score = predictOut.dataSync()[0];
  predictOut.dispose();
  setScore(score)  
  return score;
}


useEffect(()=>{
 ready().then(
    ()=>{
      loadModel(url)
      loadMetadata(url)
    }
  );

},[])

  return (

     <>
       <AppBar position="static" >
        <Toolbar >
          
          <Typography variant="h6" className={classes.title}>
            Calculate your Sentiment Score
          </Typography>
        </Toolbar>
      </AppBar>
      <Grid container style={{ height:"90vh", padding:20}}>
      <Grid item lg={6} xs={12} style={{display:"flex",alignItems:"center", flexDirection:"column"}}>
      <TextField
          id="standard-read-only-input"
          label="Type here"
          onChange={(e)=> setText(e.target.value)}
          defaultValue=""
          value={testText}
          multiline
          rows={4}
          variant="outlined"
        />
        <br/>
        <br/>
        {testText !== "" ?
              <Button Button variant="contained" color="primary" onClick={()=>getSentimentScore(testText)}>Calculate</Button>
      : <></>}

      </Grid>
<Grid item lg={6} xs ={12} style={{display:"flex", alignItems:"center", flexDirection:'column'}}>
  <br/>
<Typography>Score:</Typography>
<br/>
{testScore !==""?
<>
 <Typography style={{color:"white"}} variant="h5">{testScore}</Typography> 
 <br/>
 <Typography style={{color:"white"}}>1= Positive <br/> 0= Negative</Typography>
<br/>
<Typography>Trimmed the input text:</Typography>
<br/>
<Typography  style={{color:"white"}}> {trimedText.toString()}</Typography>
<br/>
<Typography>Map vocab to words: </Typography>
<br/>
<Typography  style={{color:"white"}}> {seqText.toString()}</Typography>
<br/>
<Typography>Fixed Input Sequence:</Typography>
<br/>
<Typography style={{color:"white",wordWrap: "break-word", width:"80%" }}>{padText.toString()}</Typography>
<br/>
<Typography>Input to 2d Tensor:</Typography>
<br/>
<Typography style={{color:"white" }}>{inputText.toString()}</Typography>
<br/>

<br/>

</>


:<></>}
 

</Grid>
      </Grid>

     </>
    
  );
}

export default App;