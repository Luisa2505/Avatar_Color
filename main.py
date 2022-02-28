# Verwendete Bibliotheken
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt


### START Rohdaten
# Messdaten einlesen
daten_csv = ['Daten/FT_20201222121753_000.csv']

daten = []
for i in range(len(daten_csv)):
    daten.append(pd.read_csv(daten_csv[i], header=0, skiprows=5, dtype='a', sep=','))

def einlesen(output, bezeichnung, datensatz):
    for i in range(len(datensatz)):
        output.append(np.zeros((len(datensatz[i][bezeichnung[0]]), len(bezeichnung))))
        for j in range(len(bezeichnung)):
            output[i][:, j] = datensatz[i][bezeichnung[j]]
        return output

rechts = []
sensor_1_23 = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22']
einlesen(rechts, sensor_1_23, daten)
rechts = np.concatenate(rechts)

links = []
sensor_24_46 = ['23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39', '40', '41', '42', '43', '44', '45']
einlesen(links, sensor_24_46, daten)
links = np.concatenate(links)

sensorCount = sensor_1_23
### ENDE Rohdaten


### START Sekundaerdaten
# Lordosewinkel bestimmen
distSensor = 20
qSensor = 0.2
expo = 0.2
rFact = 5400
measurementCnt = 1
radiusResStraight = 10000
turningPoint = 12
maxFlexion = 90
maxExtension = -90
maxRotationL = -60
maxRotationR = 60
lordosisStanding = -18
measurementFrequency = 5 # siehe csv

def calcTwistBend(rawL, rawR, sensorCount):
    radiusResL = []
    radiusResR = []
    svQuo = []
    twist = []
    bend = []
    delta = []
    alpha = []
    beta = []
    for a in range((len(sensorCount) + 2) * 2):
        delta.append(0)
        alpha.append(0)
        beta.append(0)

    for x in range(len(sensorCount)):
        if rawR[x] == 0:
            rawR[x] = 0.01
        if rawL[x] == 0:
            rawL[x] = 0.01

    for i in range(len(sensorCount)):
        rawLTemp = rawL[i]
        rawRTemp = rawR[i]

        if (rawLTemp > 0 and rawRTemp > 0) or (rawLTemp < 0 and rawRTemp < 0):
            svQuo.append(rawLTemp / rawRTemp)
        elif rawLTemp > rawRTemp or rawLTemp < rawRTemp:
            svQuo.append(np.abs(rawLTemp) + np.abs(rawRTemp))
        else:
            svQuo.append(1)

    beta[0] = np.pi

    k = 0
    for i in range(2, len(sensorCount)*2 + 1, 2):
        if svQuo[k] > 3:
            svQuo[k] = 3
        if svQuo[k] < 0.33:
            svQuo[k] = 0.33

        coeffAdd = np.abs(rawL[k] + rawR[k]) / 150
        coeffSub = np.abs(rawL[k] - rawR[k]) / 120
        coeffMax = max(coeffAdd, coeffSub)

        tmp = svQuo[k]
        if svQuo[k] > 1:
            svQuo[k] = 1 + (svQuo[k] - 1) * coeffMax
            if tmp < svQuo[k]:
                svQuo[k] = tmp
        elif svQuo[k] < 1:
            svQuo[k] = 1 - (1 - svQuo[k]) * coeffMax
            if tmp > svQuo[k]:
                svQuo[k] = tmp

        beta[i] = expo * math.log(svQuo[k])

        if rawL[k] > 0:
            beta[i] *= -1
        if beta[i] == 0:
            beta[i] = 0.001

        twist.append(beta[i] * (360 / (1 * np.pi)))
        k += 1

    for i in range(len(sensorCount)):
        if rawL[i] > 0:
            radiusL = rFact * math.pow(rawL[i], -0.9)
        else:
            radiusL = -1 * rFact * math.pow(-1 * rawL[i], -0.9)
        if rawR[i] > 0:
            radiusR = rFact * math.pow(rawR[i], -0.9)
        else:
            radiusR = -1 * rFact * math.pow(-1 * rawR[i], -0.9)

        radiusResL.append(radiusL * ((1 + qSensor) / 2 + ((1 - qSensor)/2) * math.cos(2 * (twist[i] + np.pi / 6))))
        if radiusResL[i] > radiusResStraight or radiusResL[i] < -1 * radiusResStraight:
            radiusResL[i] = 500000

        radiusResR.append(radiusR * ((1 + qSensor) / 2 + ((1 - qSensor)/2) * math.cos(2 * (twist[i] + np.pi / 6))))
        if radiusResR[i] > radiusResStraight or radiusResR[i] < -1 * radiusResStraight:
            radiusResR[i] = 500000

    k = 0
    for j in range(1, len(sensorCount) * 2 + 1, 2):
        if (radiusResL[k] >= 0 and radiusResR[k] <= 0) or (radiusResL[k]<= 0 and radiusResR[k] >= 0):
            beta[j] = 0.0001
        else:
            beta[j] = distSensor / ((radiusResL[k] + radiusResR[k]) / 2)
        bend.append(beta[j] * (360 / (2 * np.pi)))
        k += 1

    k = 0
    delta[0] = 0
    delta[1] = 0
    for j in range(2, len(sensorCount) * 2 + 2, 2):
        if (radiusResL[k] >= 0 and radiusResR[k] <= 0) or (radiusResL[k] <= 0 and radiusResR[k] >= 0):
            delta[j] = distSensor
        else:
            delta[j] = ((2 * (radiusResL[k] + radiusResR[k])) / 2) * np.sin(distSensor / ((2 * (radiusResL[k] + radiusResR[k])) / 2))
        k += 1

    for j in range(len(sensorCount) * 2, 2):
        alpha[j] = math.pi / 2
    for j in range(len(sensorCount) * 2, 2):
        alpha[i] = -math.pi / 2

    return bend, twist

coordinates = []
for i in range(len(links)):
    coordinates.append(calcTwistBend(links[i], rechts[i], sensorCount))


def separieren(input, spalte, output):
    for i in range(len(input)):
        output.append(list())
        for j in range(0, 23):
            output[i].append(input[i][spalte][j])

bend = []
twist = []
separieren(coordinates, 0, bend)
separieren(coordinates, 1, twist)

### START Funktion MovmentScore
def calcMovementScore(lastFlexion, lastTorsion, flexion, torsion):
    if lastFlexion == None or lastTorsion == None:
        lastFlexion = flexion
        lastTorsion = torsion
    flexionMovementScore = np.abs(lastFlexion - flexion) / 40
    torsionMovementScore = np.abs(lastTorsion - torsion) / 40
    lastFlexion = flexion
    lastTorsion = torsion
    movementScore = flexionMovementScore + torsionMovementScore
    return movementScore

Movementscore = []
for i in range(len(bend)-1):
    Movementscore.append(list())
    for j in range(0, 23):
        Movementscore[i].append(calcMovementScore(bend[i][j], twist[i][j], bend[i+1][j], twist[i+1][j]))                                        # for-Schleife, da keine Live-Daten

sum_ms = []
for i in range(len(Movementscore)):
    sum_ms.append((Movementscore[i][0] + Movementscore[i][1] + Movementscore[i][2] + Movementscore[i][3] + Movementscore[i][4] + Movementscore[i][5] + Movementscore[i][6] + Movementscore[i][7] + Movementscore[i][8] + Movementscore[i][9] + Movementscore[i][10] + Movementscore[i][11]))
### ENDE Funktion MovmentScore

### START Haltungspunkte bestimmen
lordosis = [0] * len(bend)
flexion = [0] * len(bend)
for i in range(len(bend)):
    for j in range(0, turningPoint):
        lordosis[i] += bend[i][j]
for i in range(len(lordosis)):
    flexion[i] = lordosis[i] - lordosisStanding

torsion = [0] * len(twist)
for i in range(len(twist)):
    for j in range(0, turningPoint):
        torsion[i] += twist[i][j]


def calcPostureScore(flexion, torsion, maxFlexion, maxExtension, maxRotationL, maxRotationR, measurementFrequency):
    scoreMatrix = [[1800, 1800, 0],
                [1800, 1440, 0],
                [1440, 1080, 0],
                [1080, 720, 0],
                [720, 360, 0],
                [360, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0]]

    if flexion >= 0:
        scoreIndexLordisis = math.ceil((flexion / maxFlexion) * 10)
    else:
        scoreIndexLordisis = math.ceil((flexion / maxExtension) * 10)

    if flexion >= 0:
        scoreIndexTorsion = math.ceil(np.abs(torsion / maxRotationL) * 2)
    else:
        scoreIndexTorsion = math.ceil(np.abs(torsion / maxRotationR) * 2)

    if scoreIndexTorsion > 2:
        scoreIndexTorsion = 2
    if scoreIndexLordisis > 10:
        scoreIndexLordisis = 10

    return scoreMatrix[scoreIndexLordisis][scoreIndexTorsion] / 3600 / measurementFrequency

scoreposture = []
for i in range(len(torsion)):
    scoreposture.append(calcPostureScore(flexion[i], torsion[i], maxFlexion, maxExtension, maxRotationL, maxRotationR, measurementFrequency))           # for-Schleife, da keine Live-Daten
### ENDE Haltungspunkte
### ENDE Sekundaerdaten     # bis hier in wurde alles aus schon vorhandenen Berechnungen genutzt

### START Daten entwickeln
sek = 60
intervall = measurementFrequency * sek
max_punkte = 100
max_ab_zu = (100 / 18000)
punkte = [max_punkte]
fak_add = [0.2, 0.4, 0.6, 0.8, 1.0]
fak_sub = [1.0, 0.7, 0.5]                               # die Parameter koennen variabel eingestellt werden

# Liste teilen in Intervalle
def split(arr, size):
    arrs = []
    while len(arr) > size:
        pice = arr[:size]
        arrs.append(pice)
        arr = arr[size:]
    arrs.append(arr)
    return arrs

if len(scoreposture) % intervall == 0:                  # Wenn die Anzahl der Eintraege ohne Rest teilbar sind, dann fuehrt er die Teilung durch
    sp_split = split(scoreposture, intervall)
    ms_split = split(sum_ms, intervall)
else:
    rest1 = len(scoreposture) % intervall
    rest2 = len(sum_ms) - (len(scoreposture) - rest1)
    for i in range(rest1):
        scoreposture.pop()
    for i in range(rest2):
        sum_ms.pop()                                # löscht die letztes ELemente, die nicht mehr ein Intervall fuellen
    sp_split = split(scoreposture, intervall)
    ms_split = split(sum_ms, intervall)             # Teilt nun die Liste in entsprechend viele Intervalle
ms_split = np.array(ms_split)
ms_split = ms_split.reshape(len(scoreposture))


mean = []
for i in range(len(sp_split)):
    mean.append(np.mean(sp_split[i]))               # Mittelwert der Listen

# rot -> 1
# gelb -> 2
# gruen -> 3

rotgelbgruen = []
for i in range(0, len(sp_split)):
    rotgelbgruen.append(list())
    for j in range(0, intervall):
        if mean[i] * 1.15 >= sp_split[i][j] >= mean[i] or mean[i] * 0.85 <= sp_split[i][j] <= mean[i]:                  # in diesem Bereich werden alle Werte als gruen -> 3 gespreichert
            rotgelbgruen[i].append(3)
        elif mean[i] * 1.3 >= sp_split[i][j] > mean[i] * 1.15 or mean[i] * 0.7 <= sp_split[i][j] < mean[i] * 0.85:      # speichert alle Werte als gelb -> 2
            rotgelbgruen[i].append(2)
        elif sp_split[i][j] > mean[i] * 1.3 or sp_split[i][j] < mean[i] * 0.7:                                          # speichert alle Werte als rot -> 1
            rotgelbgruen[i].append(1)



x = int(len(scoreposture) / len(rotgelbgruen))
rotgelbgruen_intervall = []
for i in range(len(rotgelbgruen)):
    zaehler_eins = 0
    zaehler_zwei = 0
    zaehler_drei = 0
    for j in range(intervall):
        if rotgelbgruen[i][j] == 1:
            zaehler_eins += 1
        elif rotgelbgruen[i][j] == 2:
            zaehler_zwei += 1
        elif rotgelbgruen[i][j] == 3:
            zaehler_drei += 1                                                                                           # zaehlt im Intervall wie viele Eintraege es fuer rot, gelb und gruen gibt
    if zaehler_eins >= zaehler_zwei and zaehler_eins >= zaehler_drei:
        for j in range(0, x):
            rotgelbgruen_intervall.append(1)
    elif zaehler_zwei >= zaehler_eins and zaehler_zwei >= zaehler_drei:
        for j in range(0, x):
            rotgelbgruen_intervall.append(2)
    elif zaehler_drei >= zaehler_eins and zaehler_drei >= zaehler_zwei:
        for j in range(0, x):
            rotgelbgruen_intervall.append(3)                                                                            # die haeufigsten Farbe im Intervall bestimmt fuer den Intervall die Farbe


rotgelbgruen_intervall = np.array(rotgelbgruen_intervall)
rotgelbgruen_intervall = rotgelbgruen_intervall.reshape(len(scoreposture))


for i in range(len(ms_split)):
        if ms_split[i] <= 0.5 and rotgelbgruen_intervall[i] == 1:
            if punkte[i] - fak_sub[0] * max_ab_zu <= 0:
                punkte.append(0)
            else:
                punkte.append(punkte[i] - fak_sub[0] * max_ab_zu)
        elif ms_split[i] <= 0.5 and rotgelbgruen_intervall[i] == 2:
            if punkte[i] - fak_sub[1] * max_ab_zu <= 0:
                punkte.append(0)
            else:
                punkte.append(punkte[i] - fak_sub[1] * max_ab_zu)
        elif ms_split[i] <= 0.5 and rotgelbgruen_intervall[i] == 3:
            if punkte[i] - fak_sub[2] * max_ab_zu <= 0:
                punkte.append(0)
            else:
                punkte.append(punkte[i] - fak_sub[2] * max_ab_zu)
        elif 0.5 <= ms_split[i] < 1.5:
            if punkte[i] + fak_add[0] * max_ab_zu >= 100:
                punkte.append(100)
            else:
                punkte.append(punkte[i] + fak_add[0] * max_ab_zu)                                                       # je nach Bereich wird ein Wert von den Startpunkten abgezogen
        elif 1.0 <= ms_split[i] < 1.5:
            if punkte[i] + fak_add[1] * max_ab_zu >= 100:
                punkte.append(100)
            else:
                punkte.append(punkte[i] + fak_add[1] * max_ab_zu)
        elif 1.5 <= ms_split[i] < 2.0:
            if punkte[i] + fak_add[2] * max_ab_zu >= 100:
                punkte.append(100)
            else:
                punkte.append(punkte[i] + fak_add[2] * max_ab_zu)
        elif 2.0 <= ms_split[i] < 2.5:
            if punkte[i] + fak_add[3] * max_ab_zu >= 100:
                punkte.append(100)
            else:
                punkte.append(punkte[i] + fak_add[3] * max_ab_zu)
        elif 2.5 <= ms_split[i] < 3.0:
            if punkte[i] + fak_add[4] * max_ab_zu >= 100:
                punkte.append(100)
            else:
                punkte.append(punkte[i] + fak_add[4] * max_ab_zu)                                                       # je nach Bereich wird ein Wert zu den Startpunkten hinzugefuegt

zeit = []
for i in range(len(punkte)):
    zeit.append(i/5/60)                                                                                                 # Zeitpunkte werden bestimmt (in Minuten)
### ENDE Daten entwickeln


### START Plot
plt.figure(figsize=(12, 10))
#img = plt.imread('Bild/Logo_blass.jpg')                                                                                 # Die Zeile auskommentieren, dann ist das Hintergrundbild weg
#plt.imshow(img, extent=[-15, 80, 0, 100])                                                                               # Die Zeile auskommentieren, dann ist das Hintergrundbild weg
plt.scatter(zeit, punkte, s=0.1, c=punkte, cmap='RdPu')
plt.xlabel('Zeit in [min]')
plt.ylabel('Energie als Farbsättigung in [%]')
plt.clim(0, 100)
plt.xlim(-1.0, 60.0)
plt.ylim(-1.0, 101.0)
cbar = plt.colorbar(orientation='vertical')
cbar.set_ticks([0, 100])
cbar.set_ticklabels(['0 %', '100 %'])
plt.show()

### ENDE Plot

