import pandas as pd
from sklearn.model_selection import train_test_split


def pre_process():
    '''
    sex: female or male
    age: of the patient.
    classification: covid test findings. Values 1-3 mean that the patient was diagnosed with covid in different
    degrees. 4 or higher means that the patient is not a carrier of covid or that the test is inconclusive.
    patient type: hospitalized or not hospitalized.
    pneumonia: whether the patient already have air sacs inflammation or not.
    pregnancy: whether the patient is pregnant or not.
    diabetes: whether the patient has diabetes or not.
    copd: Indicates whether the patient has Chronic obstructive pulmonary disease or not.
    asthma: whether the patient has asthma or not.
    inmsupr: whether the patient is immunosuppressed or not.
    hypertension: whether the patient has hypertension or not.
    cardiovascular: whether the patient has heart or blood vessels related disease.
    renal chronic: whether the patient has chronic renal disease or not.
    other disease: whether the patient has other disease or not.
    obesity: whether the patient is obese or not.
    tobacco: whether the patient is a tobacco user.
    usmr: Indicates whether the patient treated medical units of the first, second or third level.
    medical unit: type of institution of the National Health System that provided the care.
    intubed: whether the patient was connected to the ventilator.
    icu: Indicates whether the patient had been admitted to an Intensive Care Unit.
    death: indicates whether the patient died or recovered.
    '''
    raw_data = pd.read_csv('./data/Covid Data.csv')
    print(raw_data.head(20))
    print(raw_data.shape)

    df = raw_data.copy()

    # 各列值为 97 或 98 的，是空值，把这些值删除，因为我们显然不可能对其进行填充了
    df = df[(df.PNEUMONIA == 1) | (df.PNEUMONIA == 2)]
    df = df[(df.DIABETES == 1) | (df.DIABETES == 2)]
    df = df[(df.COPD == 1) | (df.COPD == 2)]
    df = df[(df.ASTHMA == 1) | (df.ASTHMA == 2)]
    df = df[(df.INMSUPR == 1) | (df.INMSUPR == 2)]
    df = df[(df.HIPERTENSION == 1) | (df.HIPERTENSION == 2)]
    df = df[(df.OTHER_DISEASE == 1) | (df.OTHER_DISEASE == 2)]
    df = df[(df.CARDIOVASCULAR == 1) | (df.CARDIOVASCULAR == 2)]
    df = df[(df.OBESITY == 1) | (df.OBESITY == 2)]
    df = df[(df.RENAL_CHRONIC == 1) | (df.RENAL_CHRONIC == 2)]
    df = df[(df.TOBACCO == 1) | (df.TOBACCO == 2)]

    # 男性无法进行生育
    df.PREGNANT = df.PREGNANT.replace(98, 2)
    df.PREGNANT = df.PREGNANT.replace(97, 2)

    # INTUBED和ICU含有太多的空值，所以直接删除这两列
    df.drop("INTUBED", axis=1, inplace=True)
    df.drop("ICU", axis=1, inplace=True)

    # CLASIFFICATION_FINAL中[1,2,3]代表感染，[4,5,6,7]代表未感染
    df.CLASIFFICATION_FINAL = df.CLASIFFICATION_FINAL.replace([1, 2, 3], 1)
    df.CLASIFFICATION_FINAL = df.CLASIFFICATION_FINAL.replace([4, 5, 6, 7], 2)

    df.DATE_DIED = df.DATE_DIED.replace('9999-99-99', 0)
    df.loc[df['DATE_DIED'] != 0, 'DATE_DIED'] = 1

    df['SURVIVAL'] = df.DATE_DIED
    df.drop('DATE_DIED', axis=1, inplace=True)

    df.drop('CLASIFFICATION_FINAL', axis=1)

    y = df["SURVIVAL"]
    x = df.drop("SURVIVAL", axis=1)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

    return x_train, x_test, y_train, y_test
