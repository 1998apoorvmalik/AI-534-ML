#!/usr/bin/env python3

__author__ = "Liang Huang"

'''
Linux/Mac: cat  income.test.blind | python3 random.py > income.test.predicted
Windows:   type income.test.blind | python3 random.py > income.test.predicted

Note: Windows command-line uses "type" for the Unix/Linux command of "cat", which displays the content of a file.
      Also note that pipes ("... | ...") and I/O re-direction ("... > ...") works perfectly on Windows.
      This script takes standard input and writes to standard output. 
      Alternatively, you can also use file input and file output, as TA Sizhen demonstrated on Slack, e.g.:

       with open("income.test.predicted", "w") as f:
         for entry in output:
           s = ", ".join(entry)
           f.write("%s\n" % s)

      But I think standard I/O is easier since you don't have the notion of "file" in your code -- instead, file I/O is handled by the command-line.

      This code works for both Python 3 and Python 2, but Python 3 is recommended.

      You can also use this code together with validate.py, and will get something like:

      cat income.test.blind | python3 random_output.py | python3 validate.py

      Your positive rate is 51.0%.
      ERROR: Your positive rate seems too high (should be similar to train and dev).
      PLEASE CHECK YOUR CODE AND REFER TO income.dev.txt FOR THE FORMAT.

      Please learn the three very useful string functions from this code: strip(), split(), and join().
'''

import sys
import random

for line in sys.stdin:
    fields = line.strip().split(", ") # extract the 9 input fields; call strip() to remove the final "\n"
    # ... normally, you should do k-NN calculations on features from these fields ...
    # ... but instead, let's just use random here ...
    label = ">50K" if random.random() > 0.5 else "<=50K" # random.random() returns a real number in the range of [0, 1)
    print(", ".join(fields + [label])) # output 10 fields, separated by ", "
