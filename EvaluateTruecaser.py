from Truecaser import *
import pickle
import nltk
import string


def evaluateTrueCaser(testSentences, wordCasingLookup, uniDist, backwardBiDist, forwardBiDist, trigramDist):
    correctTokens = 0
    totalTokens = 0
    
    for sentence in testSentences:
        tokensCorrect = nltk.word_tokenize(sentence)
        tokens = [token.lower() for token in tokensCorrect]
        tokensTrueCase = getTrueCase(tokens, 'title', wordCasingLookup, uniDist, backwardBiDist, forwardBiDist, trigramDist)
        
        perfectMatch = True
        
        for idx in range(len(tokensCorrect)):
            totalTokens += 1
            if tokensCorrect[idx] == tokensTrueCase[idx]:
                correctTokens += 1
            else:
                perfectMatch = False
        
        if not perfectMatch:
            print(tokensCorrect)
            print(tokensTrueCase)
        
            print("-------------------")
    

    print("Accuracy: %.2f%%" % (correctTokens / float(totalTokens)*100))
    
    
def defaultTruecaserEvaluation(wordCasingLookup, uniDist, backwardBiDist, forwardBiDist, trigramDist):
    testSentences = [
        "Ross Perot wäre vielleicht ein prächtiger Diktator",
        "Konzernchefs lehnen den Milliardär als US-Präsidenten ab",
        "Texaner gibt nur vage Auskunft über seine Wirtschaftspolitik",
        "Der texanische Milliardär Ross Perot hat das politische Establishment in Washington aufgeschreckt",
        "Nach Meinungsumfragen liegt der parteilose Self-mademan gut im Rennen um den Chefsessel im Weißen Haus mit dem amtierenden Präsidenten George Bush und dem Demokraten Bill Clinton .",
        "Daß Perot ein Unternehmen erfolgreich leiten kann , davon sind selbst seine Kritiker überzeugt .",
        "Ob diese Fähigkeiten aber ausreichen , um die größte Volkswirtschaft der Welt aus ihrer Krise zu führen , bezweifeln viele Ökonomen .",
        "Und auch die Konzernchefs in den USA halten nicht viel von dem 62jährigen .",
        "Zwar können sich die meisten Topmanager durchaus einen Unternehmer als Präsidenten vorstellen - nur nicht ausgerechnet Perot .",
        "Nach einer Umfrage des Wirtschaftsmagazins Fortune unter den Bossen von 500 Großunternehmen wünschten im Mai nur elf Prozent `` Ross for President '' , während 78 Prozent sich für Bush und vier Prozent für Clinton aussprachen .",
        "Allerdings glaubt fast die Hälfte der Chief Executives , daß Perot durchaus Chancen habe , die Wahl im November zu gewinnen , wenn er denn kandidiert .",
        "Als größte Schwäche des Texaners nennen die Befragten seinen Mangel an Erfahrung auf dem politischen Parkett .",
        "Viele meinen , daß Perot mit seinem Befehlston auf dem Capitol gegen eine Wand laufen würde .",
        "So erklärt etwa Edward Brandon von dem Unternehmen National City in Ohio :",
        "`` Ich glaube kaum , daß mit seinem , naja , etwas undiplomatischen Stil im Weißen Haus dem Land ein Gefallen getan wäre .",
        "Er wäre vielleicht ein prächtiger Diktator - aber das ist nicht unser System . ''",
        "Und ein anderer Manager vermutet , daß sich `` ein Dogmatiker wie Perot in Washington schwer tun würde , es sei denn er schafft den Kongreß ab '' .",
        "Ein ehemaliger Geschäftsführer , der heute in fünf Konzernen im Aufsichtsgremium sitzt , `` kennt niemanden , der nicht glaubt , daß Perot als Präsident eine absolute Katastrophe wäre '' .",
        "Allerdings gibt es dem Magazin zufolge in kleinen und mittleren Firmen viele Unternehmer , die meinen , Perot sei einer von ihnen , und die den Texaner unterstützen .",
        "Zwei Themen , die Perot immer wieder anspricht , Rezession und Bürokratie , machen ihnen besonders zu schaffen .",
        "Einer , der sich für den Milliardär ausspricht , ist Steve Jobs , dem Perot für den Aufbau der Computerfirma Next 20 Millionen Dollar bereitstellte .",
        "Die Spekulationen darüber , welche Art Präsident `` Henry der Hammer '' , so ein Spitzname aus Perots Jugendzeit , wäre , orientieren sich daran , wie Perot seine Milliarden machte .",
        "Die Story geht so :",
        "Mit 1000 von seiner Frau geborgten Dollar gründet der ehemalige IBM-Verkäufer 1962 die Electronic Data Services ( EDS ) und zeigt damit dem Rechnergoliath , daß es nicht nur gilt , Computer abzusetzen , sondern auch Geld damit zu verdienen ist , wenn man den Benutzern beim Einsatz der Maschinen hilft .",
        "Das Informatik-Dienstleistungsunternehmen verkauft er 1984 für 2,5 Milliarden an General Motors .",
        "Er tritt in die GM-Verwaltung ein und wird Großaktionär des Autokonzerns .",
        "Ihm gelingt es aber nicht , den Koloß , der der Regierung in Washington mehr ähnelt als jeder andere Konzern , in Schwung zu bringen .",
        "Nach zwei Jahren scheidet er dort aus , mit einer zusätzlichen GM-Zahlung von 700 Millionen Dollar in der Tasche .",
        "Heutzutage hält Perot sein Vermögen vor allem in Staatspapieren und Immobilien .",
        "Außerdem gehören ihm der Computer-Dienstleister Perot Systems , Beteiligungen an kleineren Firmen und persönliche Reichtümer wie Jachten , Jets und Villen .",
        "Von modernem Management hat Perot , so Fortune , nie etwas gehalten .",
        "Für ihn zählte allein `` Führerschaft '' .",
        "Von seinen Beschäftigten verlange er vor allem Arbeitsmoral und ordentliches Auftreten .",
        "Perot sei ein autoritärer Macher , und das sei auch seine Schwäche , beschreibt ihn ein Manager .",
        "Die Kritik der Ökonomenzunft an Perot entzündet sich vor allem an dessen vagen Äußerungen zur Wirtschaftspolitik .",
        "`` Es ist wirklich schwer zu sagen , welche Positionen er einnimmt , da er sich noch nicht konkret geäußert hat '' , beklagen Volkswirte .",
        "Bisher bestehe sein `` Programm '' nur aus Allgemeinplätzen .",
        "So will der politische Außenseiter beispielsweise das Steuersystem vereinfachen , das Bildungssystem verbessern , das gigantische Haushaltsdefizit abbauen , Einfuhren aus Japan drosseln und die geplante Freihandelszone der USA mit Mexiko verhindern .",
        "Wie er diese Aufgaben lösen will , verrät er aber nicht .",
        "Statt Details zu nennen , wiederholt er unverdrossen die `` Erfolgsformel '' :",
        "Die Amerikaner müssen hart arbeiten , um das Land wieder auf den richtigen Kurs zu bringen .",
        "`` Die Probleme unseres Landes sind doch weit größer als die eines Unternehmens '' , meint Rudy Oswald , Chefvolkswirt beim Gewerkschaftsbund AFL-CIO .",
        "Schließlich sei die Regierung auch für die nationale Sicherheit und allgemeine Wohlfahrt verantwortlich .",
        "`` Hier geht es um mehr , als nur Profit zu machen '' , fügt er hinzu .",
        "`` Geschäftemachen ist seine Welt und nicht die Politik .",
        "Er ist ein eigenwilliger Einzelgänger , der sich zudem nicht an Spielregeln hält '' , zeigt sich ein anderer Fachmann skeptisch .",
        "Früher oder später , da sind sich alle einig , muß Perot Farbe bekennen und Konzepte vorlegen .",
        "`` Die Frage ist nur '' , meint ein Finanzexperte , `` ob er ins Weiße Haus einziehen kann , ohne uns vorher zu sagen , was er eigentlich machen will . ''"
    ]
    
    evaluateTrueCaser(testSentences, wordCasingLookup, uniDist, backwardBiDist, forwardBiDist, trigramDist)
    
if __name__ == "__main__":       
    with open('distributions.obj', 'rb') as f:
        uniDist = pickle.load(f)
        backwardBiDist = pickle.load(f)
        forwardBiDist = pickle.load(f)
        trigramDist = pickle.load(f)
        wordCasingLookup = pickle.load(f)    
    defaultTruecaserEvaluation(wordCasingLookup, uniDist, backwardBiDist, forwardBiDist, trigramDist)
