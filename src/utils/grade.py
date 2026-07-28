def compute_grade(eer):
    if eer > 10.9:
        grade = 0
    elif eer < 5.3:
        grade = 10
    else:
        # Linear interpolation between 2 and 10
        grade = 2 + (10.9 - eer) * (8 / (10.9 - 5.3))
    return grade
