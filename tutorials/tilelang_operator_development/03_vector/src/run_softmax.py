"""Validate the five complete outputs of the three register-based Softmax exercises."""
import argparse
from softmax_kernel import softmax_single, softmax_kernel, softmax_pipeline
from course_validation import cpu_input, checked_input, check_softmax

def run(level="all"):
    configurations = {
        "easy": [("single", 8, 128, lambda: softmax_single(8,128))],
        "medium": [("serial blocks=1",9216,128,lambda:softmax_kernel(9216,128,8,1)),
                   ("serial blocks=36",9216,128,lambda:softmax_kernel(9216,128,8,36))],
        "hard": [("pipeline stages=2",9216,128,lambda:softmax_pipeline(9216,128,8,36,2)),
                 ("pipeline stages=3",9216,128,lambda:softmax_pipeline(9216,128,8,36,3))]}
    for group in configurations if level=="all" else [level]:
        for name,M,C,factory in configurations[group]:
            kernel=factory()
            for case in ("random","zero","constant","large"):
                host=cpu_input((M,C),case);x=checked_input(host)
                check_softmax(kernel(x),host,f"{group} {name} {case}")
if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("level",nargs="?",default="all",choices=["all","easy","medium","hard"])
    run(parser.parse_args().level)
