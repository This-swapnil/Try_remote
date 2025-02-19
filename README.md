<<<<<<< HEAD
# Try_remote
=======
# LLM-RAG-WebUI-integration

## Important note
* You must code in Python.
* Some of the libraries and dependencies to fulfill this problem may require python version of atleast 3.8 to 3.10 I think ( I only tried Python 3.8), I don't think python 3.11 works with some dependencies. This is just my opinion, you can try out different python environments using conda or python venv. Ideally please use conda.
* The "bare-minimum" task can be accomplished with just CPU. If your GPU is good enough just use that though.
* This task will require you to run your code from your machines OS terminal. For windows, its the powershell (there's another shell too I think), for macOS and Linux machines a common terminal is bash. Your OS might be using a different terminal from what I mentioned, or might have multiple, doesn't matter, just use one.
* After completing this task you will need to screen record to make a video showing that your code works and you explaining how it works. Obviously in the screen recording you MUST run your program from the terminal.
* Create a github repo containing your code and the video. Name the repo something like "JSB_interview_problem" or something like that so it's identifiable.
* **IMPORTANT (READ THE ENTIRE BULLET POINT) For submission you must:**
   - **submit a pull request to this repo so that we have access to your username and get find your repo. However, so others don't copy your work, do not do you work in the public forked repo.**
   - **Make a private clone (or however you make things private) of the forked repo and do your actual work there.**
   - **Send an invite to me to the private repo (philipamadasun1@gmail.com) so I can gain access.**
   - **Please don't make me have to remake this repo again. In your ReadME, make sure to provide your email address.**
* You may freely use any tool available to you to accomplish this task. The internet, ChatGPT, anything (**EXCEPT stuff like co-pilot**), just get it done.
  

## Problem task
Our lab works with Large Language Models (LLMs) and other transformer based models in a social robotics research (both applied and fundamental). We are looking for people who are either familair with these topics or can quickly familiarize themselves. You are to create software to communicate with an LLM to perform different tasks:
* Back and forth communication. Which means, you provide a user query like "hey what's up!", and the LLM replies with an output: "not much just here to assist you". This will be the primary function of the software. Create a system prompt for the LLM to follow, it  can be anything you want. 
* The software must have a mode where it performs Retreival Augmented Generation (RAG). This way, one can provide the LLM with articles from links and pdf-links. This way the user can ask the LLM about information from these articles. You may use these: [blog](https://ollama.com/blog/run-llama2-uncensored-locally) and [pdf-link](https://d18rn0p25nwr6d.cloudfront.net/CIK-0001813756/975b3e9b-268e-4798-a9e4-2a9a7c92dc10.pdf)
* **For clarity, the software MUST have the ability to switch in and out of both modes without having to restart the software.**
* Software displays both user queries and LLM responses in a WebUI. That is to say the webUI does not show up empty or restart, after every mode switch. user query and LLM response must be different colors on the GUI. So either color of text or word-bubble must be different.
* **Note** Inbetween modes make sure the LLM is keeping context of the conversation so it remembers the whole conversation. For the tiniest LLMs it might be a bit rough as you only have context length of a little over 2000. If you're in this situation, the conversation in your video demo should not be too long.

## Tools you can use
The platform I advise to run LLMs from is [ollama](https://ollama.com/), here is their [repo](https://github.com/ollama/ollama). The ollama repo also provides some example scripts that might provide some inspiration on how to go about solving some parts of the problem.
For those with not so good PCs, again the "bare-minimum" can be done with just CPU, you can pull a small LLM like `gemma:2b` or `tinyllama` (these are around 2GB in size) locally on your ollama and just use those. For the webUI you may use streamlit and Flask as a server to retreive user queries and LLM responses from. I have provided two scripts which use streamlit and Flask to show a simple example of to get user input to show up on the streamlit webUI. Again, this is just advice, any other way you can get this done, you can just do that. You don't have to use ollama , or streamlit or Flask.

## "bare-minimum"
* You can provide the LLM with user query by typing it in **terminal**.
   - Please make sure you understand what I am asking you to do. Do not make a progam where you input queries in the webUI.
   - You input queries and change modes straight in your terminal, not the webUI.
* You can provide some kind of trigger word or phrase in terminal to switch modes
* use tiny models like gemma (2 billion parameters) or tiny llama (some 1 billion parameters or so)
* And of course your software must do everything the problem task requires.


## Extra Credit
Those who are able to go some extra mile will be picked first for interview. Have your software have these extra capabiltiies. These extra capabilities should not be confused with modes. The software should have this capability from any mode.

* Talk to your computer instead of typing in terminal. Here are some tools you may need:
  * A tool to stream audio. You can use `pyaudio` or some other library.
  * A tool to detect human speech in audio, so a Voice Activity Detector (`webrtcvad`, `sileroVAD` etc.).
  * A tool to derive text from speech, so maybe a  transcriber model (i.e whisper, faster-whisper etc.). Even with a "bare-minimum" PC, some of these transcriber models can run on only CPU.
* Your machine talks back
    * You will need some text-to-speech library or model for the program to talk back as a bonus. The voice doesn't have to be great. 
* If you have a decent enough PC:
    * Use bigger LLMs like mistral (fine-tuned instruct version)
    * You can also create a mode that uses a multimodal model like Llava.


## Contact
Email: philipamadasun1@gmail.com
>>>>>>> myrepo/main
