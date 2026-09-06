IMPORTANT: to run the email feature, you have to create a .env file in this folder with the following keys:
-> EMAIL 
-> PASSWD
-> SERVER

To produce useful output, the 'Prospector code.py' file must be run with keywords:

--showplots                 to show plots.
--readfile [FILE_PATH]      OPTIONAL: specify .h5 file to plot
--fitnewmodel               to fit a new model (takes a while!).
--outfile [FILE_PATH]       specify .h5 file to save to
--minimize                  to perform optimisation before fitting (not that useful).