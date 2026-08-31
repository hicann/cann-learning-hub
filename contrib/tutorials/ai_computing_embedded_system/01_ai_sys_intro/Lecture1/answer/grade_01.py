import os

def grade(user_globals):
    notebook_dir = os.getcwd()
    candidates = [
        os.path.join(notebook_dir, 'answer', '01_answer.txt'),
        os.path.join(notebook_dir, '01_answer.txt'),
        os.path.join(os.path.dirname(__file__), '01_answer.txt'),
    ]
    answer_file = None
    for c in candidates:
        if os.path.exists(c):
            answer_file = c
            break
    if answer_file is None:
        print('❌ 找不到答案文件 01_answer.txt')
        return

    correct = {}
    with open(answer_file) as f:
        for line in f:
            line = line.strip()
            if line:
                num, ans = line.split('.', 1)
                correct[int(num.strip())] = ans.strip().upper()

    my_answers = {}
    unanswered = []
    for q in sorted(correct.keys()):
        var = f'q{q}'
        if var in user_globals:
            val = user_globals[var]
            ans = val.strip().upper() if isinstance(val, str) else str(val).strip().upper()
            if not ans:
                unanswered.append(q)
            my_answers[q] = ans
        else:
            unanswered.append(q)
            my_answers[q] = ''

    if unanswered:
        print(f'⚠️ 以下题目尚未作答：{unanswered}')
        print('请填入答案并运行对应单元格后，再运行批改。')
        return

    wrong = []
    for q in sorted(correct.keys()):
        if my_answers.get(q, '') != correct[q]:
            wrong.append(q)

    if not wrong:
        print('🎉 全部正确！')
    else:
        print(f'以下题目答错：{wrong}')
        print(f'得分：{len(correct) - len(wrong)}/{len(correct)}')
