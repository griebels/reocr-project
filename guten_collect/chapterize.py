'''
  chapterize by code: https://github.com/JonathanReeve/chapterize
'''

import click
import logging
import re
import os


class Book():
    def __init__(self, contents, stats=False):
        self.contents = contents
        self.lines = self.getLines()
        self.headings = self.getHeadings()

        if len(self.headings) > 0:
            # Alias for historical reasons. FIXME
            self.headingLocations = self.headings
            self.ignoreTOC()
            #logging.info('Heading locations: %s' % self.headingLocations)
            self.heading_str = [self.lines[loc].lstrip().rstrip() for loc in self.headingLocations]
            #logging.info('Headings: %s' % headingsPlain)
            self.chapters = self.getTextBetweenHeadings()
            # logging.info('Chapters: %s' % self.chapters)
            self.numChapters = len(self.chapters)

        # if stats:
        #     self.getStats()
        # else:
        #     self.writeChapters()


    def getLines(self):
        """
        Breaks the book into lines.
        """
        return self.contents.split('\n')

    def getHeadings(self):

        # Form 1: Chapter I, Chapter 1, Chapter the First, CHAPTER 1
        # Ways of enumerating chapters, e.g.
        arabicNumerals = '\d+'
        romanNumerals = '(?=[MDCLXVI])M{0,3}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})'
        numberWordsByTens = ['twenty', 'thirty', 'forty', 'fifty', 'sixty',
                              'seventy', 'eighty', 'ninety']
        numberWords = ['one', 'two', 'three', 'four', 'five', 'six',
                       'seven', 'eight', 'nine', 'ten', 'eleven',
                       'twelve', 'thirteen', 'fourteen', 'fifteen',
                       'sixteen', 'seventeen', 'eighteen', 'nineteen'] + numberWordsByTens
        numberWordsPat = '(' + '|'.join(numberWords) + ')'
        ordinalNumberWordsByTens = ['twentieth', 'thirtieth', 'fortieth', 'fiftieth', 
                                    'sixtieth', 'seventieth', 'eightieth', 'ninetieth'] + \
                                    numberWordsByTens
        ordinalNumberWords = ['first', 'second', 'third', 'fourth', 'fifth', 'sixth', 
                              'seventh', 'eighth', 'ninth', 'twelfth', 'last'] + \
                             [numberWord + 'th' for numberWord in numberWords] + ordinalNumberWordsByTens
        ordinalsPat = '(the )?(' + '|'.join(ordinalNumberWords) + ')'
        enumeratorsList = [arabicNumerals, romanNumerals, numberWordsPat, ordinalsPat] 
        enumerators = '(' + '|'.join(enumeratorsList) + ')'
        form1 = 'chapter ' + enumerators

        # # Form 2: II. The Mail
        # enumerators = romanNumerals
        # separators = '(\. | )'
        # titleCase = '[A-Z][a-z]'
        # form2 = enumerators + separators + titleCase


        # Form 3: II. THE OPEN ROAD
        enumerators = romanNumerals
        separators = '(\.\s+)'
        titleCase = '[A-Z][A-Z]'
        form3 = enumerators + separators + titleCase

        # Form 4: INTRODUCTION
        form4 = 'INTRODUCTION' + '(' + '|'.join([ ':', '.']) + ')'

        # Form 5: FOOTNOTES
        form5 = '(FOOTNOTE)(S\b)?' + '(' + '|'.join(['', ':']) + ')'
        form6 = '\[APPENDIX (FOOTNOTE)(S\b)\]'

        # Form 7: Contents
        form7 = '(CONTENT)(S\b)?' + '(' + '|'.join(['', ':', '.']) + ')'

        # Form 8: Transcriber’s Notes
        form8 = '(' + '|'.join(['', '  ']) + ')' + 'Transcriber' + '(' + '|'.join(['’', '\'']) + ')' + 's (Note)(s\b)?' + '(' + '|'.join(['', ':', '.']) + ')'

        # Form 9: INDEX
        form9 = '(' + '|'.join(['', '  ']) + ')' + 'INDEX' + '(' + '|'.join(['', ':', '.']) + ')'

        # Form 10: Publisher's Explanatory Note
        form10 = 'Publisher\'s Explanatory Note' + '(' + '|'.join(['', ':', '.']) + ')'
        
        # Form 11: THE END.
        form11 = 'THE END' + '(' + '|'.join(['', '.']) + ')'

        # # Form 4: a number on its own, e.g. 8, VIII
        # arabicNumerals = '^\d+\.?$'
        # romanNumerals = '(?=[MDCLXVI])M{0,3}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})\.?$'
        # enumeratorsList = [arabicNumerals, romanNumerals]
        # enumerators = '(' + '|'.join(enumeratorsList) + ')'
        # form4 = enumerators

        pat = re.compile(form1, re.IGNORECASE)
        # This one is case-sensitive.
        #pat2 = re.compile('(%s|%s|%s)' % (form2, form3, form4))
        pat2 = re.compile('(%s|%s|%s|%s|%s|%s|%s|%s|%s)' % (form3, form4, form5, form6, form7, form8, form9, form10, form11))

        # TODO: can't use .index() since not all lines are unique.
        
        #headings_str = []                           #edit by MJ
        headings = []
        for i, line in enumerate(self.lines):
            if pat.match(line.lstrip()) is not None:
                headings.append(i)
                #headings_str.append(line)           #edit by MJ
            if pat2.match(line.lstrip()) is not None:
                headings.append(i)
                #headings_str.append(line)           #edit by MJ

        if len(headings) < 3:                       #edit by MJ
            # logging.info('Headings: %s' % headings)
            # logging.error("Detected fewer than three chapters. This probably means there's something wrong with chapter detection for this book.")
            # exit()
            headings = []
            #headings_str = []   
            return headings#, headings_str

        self.endLocation = self.getEndLocation()

        # Treat the end location as a heading.
        headings.append(self.endLocation)
        #headings_str.append('END')                  #edit by MJ

        return headings#, headings_str

    def ignoreTOC(self):
        """
        Filters headings out that are too close together,
        since they probably belong to a table of contents.
        """
        pairs = zip(self.headingLocations, self.headingLocations[1:])
        toBeDeleted = []
        for pair in pairs:
            delta = pair[1] - pair[0]
            if delta < 4:
                if pair[0] not in toBeDeleted:
                    toBeDeleted.append(pair[0])
                if pair[1] not in toBeDeleted:
                    toBeDeleted.append(pair[1])
        logging.debug('TOC locations to be deleted: %s' % toBeDeleted)
        for badLoc in toBeDeleted:
            index = self.headingLocations.index(badLoc)
            del self.headingLocations[index]

    def getEndLocation(self):
        """
        Tries to find where the book ends.
        """
        ends = ["End of the Project Gutenberg EBook",
                "End of Project Gutenberg's",
                "\*\*\*END OF THE PROJECT GUTENBERG EBOOK",
                "\*\*\* END OF THIS PROJECT GUTENBERG EBOOK",
                "\*\*\* END OF THIS PROJECT GUTENBERG EBOOK DAISY THORNTON \*\*\*"]   #edited by MJ
        joined = '|'.join(ends)
        pat = re.compile(joined, re.IGNORECASE)
        endLocation = None
        for line in self.lines:
            if pat.match(line.strip()) is not None:          #edited by MJ
                endLocation = self.lines.index(line)
                self.endLine = self.lines[endLocation]
                break

        if endLocation is None: # Can't find the ending.
            logging.info("Can't find an ending line. Assuming that the book ends at the end of the text.")
            endLocation = len(self.lines)-1 # The end
            self.endLine = 'None'

        logging.info('End line: %s at line %s' % (self.endLine, endLocation))
        return endLocation

    def getTextBetweenHeadings(self):
        chapters = []
        lastHeading = len(self.headingLocations) - 1
        for i, headingLocation in enumerate(self.headingLocations):
            if i != lastHeading:
                nextHeadingLocation = self.headingLocations[i+1]
                chapters.append(self.lines[headingLocation+1:nextHeadingLocation])
            
        return chapters

    def zeroPad(self, numbers):
        """
        Takes a list of ints and zero-pads them, returning
        them as a list of strings.
        """
        maxNum = max(numbers)
        maxDigits = len(str(maxNum))
        numberStrs = [str(number).zfill(maxDigits) for number in numbers]
        return numberStrs

    def getStats(self):
        """
        Returns statistics about the chapters, like their length.
        """
        # TODO: Check to see if there's a log file. If not, make one.
        # Write headings to file.
        numChapters = self.numChapters
        averageChapterLength = sum([len(chapter) for chapter in self.chapters])/numChapters
        headings = ['Filename', 'Average chapter length', 'Number of chapters']
        stats = ['"' + self.filename + '"', averageChapterLength, numChapters]
        stats = [str(val) for val in stats]
        headings = ','.join(headings) + '\n'
        statsLog = ','.join(stats) + '\n'
        logging.info('Log headings: %s' % headings)
        logging.info('Log stats: %s' % statsLog)

        if not os.path.exists('log.txt'):
            logging.info('Log file does not exist. Creating it.')
            with open('log.txt', 'w') as f:
                f.write(headings)
                f.close()

        with open('log.txt', 'a') as f:
            f.write(statsLog)
            f.close()

    def writeChapters(self):
        chapterNums = self.zeroPad(range(1, len(self.chapters)+1))
        logging.debug('Writing chapter headings: %s' % chapterNums)
        basename = os.path.basename(self.filename)
        noExt = os.path.splitext(basename)[0]

        if self.nochapters:
            # Join together all the chapters and lines.
            text = ""
            for chapter in self.chapters:
                # Stitch together the lines.
                chapter = '\n'.join(chapter)
                # Stitch together the chapters.
                text += chapter + '\n'
            ext = '-extracted.txt'
            path = noExt + ext
            with open(path, 'w') as f:
                f.write(text)
        else:
            logging.info('Filename: %s' % noExt)
            outDir = noExt + '-chapters'
            if not os.path.exists(outDir):
                os.makedirs(outDir)
            ext = '.txt'
            for num, chapter in zip(chapterNums, self.chapters):
                path = outDir + '/' + num + ext
                logging.debug(chapter)
                chapter = '\n'.join(chapter)
                with open(path, 'w') as f:
                    f.write(chapter)