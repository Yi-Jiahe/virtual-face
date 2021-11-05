# Note on requirements.txt
[dlib](https://github.com/davisking/dlib#compiling-dlib-python-api) needs to be compiled during its installation, which 
requires some c tooling to be avaliable on your computer. This [article](https://towardsdatascience.com/face-landmark-detection-using-python-1964cb620837#:~:text=let%E2%80%99s%20get%20started!-,Face%20Landmark%20Detection%20with%20Dlib,-Dlib%20is%20a) 
provides a simple setup.

Also note that the setup.py step of the dlib install takes a while and uses quite a bit of memory. Use `pip install requirements.txt -vvv` 
for assurances that its still doing something and give it about 10-15 minutes to complete.