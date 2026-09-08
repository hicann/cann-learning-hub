"""
grade_07.py - 实验4.5 课后练习批改脚本
实验4.5 昇腾香橙派部署深度学习网络实验
"""


def grade(g):
    """批改实验4.5的10道课后练习题"""
    answers = {
        'q1':  'C',
        'q2':  'B',
        'q3':  'B',
        'q4':  'B',
        'q5':  'B',
        'q6':  'B',
        'q7':  'C',
        'q8':  'B',
        'q9':  'B',
        'q10': 'B',
    }

    score = 0
    total = len(answers)

    print("=" * 50)
    print("         实验4.5 批改结果")
    print("=" * 50)

    for qid, correct_ans in answers.items():
        student_ans = g.get(qid, '').strip().upper()
        if student_ans == correct_ans:
            print(f"  {qid}: {student_ans}  正确")
            score += 1
        elif student_ans:
            print(f"  {qid}: {student_ans}  错误 (正确答案: {correct_ans})")
        else:
            print(f"  {qid}: 未作答 (正确答案: {correct_ans})")

    print("=" * 50)
    print(f"  得分: {score}/{total}  ({score/total*100:.0f}%)")
    if score == total:
        print("  全部正确! 恭喜你掌握了昇腾香橙派部署深度学习的完整流程!")
    elif score >= total * 0.6:
        print("  及格，建议复习错题相关内容")
    else:
        print("  未及格，请重新学习实验内容")
    print("=" * 50)
