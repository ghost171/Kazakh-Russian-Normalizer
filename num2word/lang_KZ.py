# -*- coding: utf-8 -*-
# Copyright (c) 2003, Taro Ogawa.  All Rights Reserved.
# Copyright (c) 2013, Savoir-faire Linux inc.  All Rights Reserved.

# This library is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation; either
# version 2.1 of the License, or (at your option) any later version.
# This library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# Lesser General Public License for more details.
# You should have received a copy of the GNU Lesser General Public
# License along with this library; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
# MA 02110-1301 USA

from __future__ import unicode_literals

from .base import Num2Word_Base
from .utils import get_digits, splitbyx

import numpy as np

ZERO = 'нөл'

ONES = {
    1: 'бір',
    2: 'екі',
    3: 'үш',
    4: 'төрт',
    5: 'бес',
    6: 'алты',
    7: 'жеті',
    8: 'сегіз',
    9: 'тоғыз',
}

FRACTION = {
    1: 'оннан',
    2: 'жүзден',
    3: 'мыңнан',
    4: 'он мыңнан',
    5: 'жүз мыңнан',
    6: 'миллионнан',
    7: 'он миллионнан',
    8: 'жүз миллионнан',
    9: 'миллиардтан',
    10: 'он миллиардтан',
    11: 'жүз миллиардтан',
    12: 'триллионтан',
    13: 'он триллионтан',
    14: 'жүз триллионтан',
    15: 'квадриллионтан',
    16: 'он квадриллионтан',
    17: 'жүз квадриллионтан',
    18: 'квинтиллионтан',
    19: 'он квинтиллионтан',
    20: 'жүз квинтиллионтан'
}

TEN = 'он'

TWENTIES = {
    2: 'жиырма',
    3: 'отыз',
    4: 'қырық',
    5: 'елу',
    6: 'алпыс',
    7: 'жетпіс',
    8: 'сексен',
    9: 'тоқсан',
}

HUNDRED = 'жүз'

THOUSANDS = {
    1: 'мың',
    2: 'миллион',
    3: 'миллиард',
    4: 'триллион',
    5: 'квадриллион',
    6: 'квинтиллион',
    7: 'секстиллион',
    8: 'септиллион',
    9: 'октиллион',
    10: 'нониллион',
}

ORDINALS = {
            "нөл": "нөлінші",
            "бір": "бірінші",
            "екі": "екінші",
            "үш": "үшінші",
            "төрт": "төртінші",
            "бес": "бесінші",
            "алты": "алтыншы",
            "жеті": "жетінші",
            "сегіз": "сегізінші",
            "тоғыз": "тоғызыншы",
            "он": "оныншы",
            "жиырма": "жиырмасыншы",
            "отыз": "отызыншы",
            "қырық": "қырықыншы",
            "елу": "елуінші",
            "алпыс": "алпысыншы",
            "жетпіс": "жетпісінші",
            "сексен": "сексенінші",
            "тоқсан": "тоқсаныншы",
            "жүз": "жүзінші",
            "мың": 'мыңыншы',
            "миллион": "миллионыншы",
            "миллиард": "миллиардыншы",
            "триллион": "триллионыншы",
            "квадриллион": "квадриллионыншы",
            "квинтиллион": "квинтиллионыншы",
            "секстиллион": "секстиллионыншы",
            "септиллион": "септиллионыншы",
            "октиллион": "октиллионыншы",
            "нониллион": "нониллионыншы",
        }

class Num2Word_KZ(Num2Word_Base):
    CURRENCY_FORMS = {
        'USD': ('доллар', 'цент'),
        'KZT': ('теңге', 'тиын'),
    }

    def setup(self):
        self.negword = "минус"
        self.pointword = "бүтін"
        self.errmsg_nonnum = "type(%s) not in [long, int, float]"
        self.errmsg_floatord = "Cannot treat float %s as ordinal."
        self.errmsg_negord = "Cannot treat negative num %s as ordinal."
        self.errmsg_toobig = "abs(%s) must be less than %s."

    def to_cardinal(self, number):
        # try:
        #    float(number) == number
        # except (ValueError, TypeError, AssertionError, AttributeError):
        #     raise TypeError(self.errmsg_nonnum % number)
        n = str(number).replace(',', '.')
        #print("n", n)
        # print("KKKKKKK")

        if ('e' in n):
            n = ("%.9f" % float(number))

        n = str(n).replace(',', '.')

        if '.' in n:
            left, right = n.split('.')
            right_converted = right

            for i in range(len(right) - 1, 0, -1):
                if right[i] != '0':
                    break
                right_converted = right_converted[:len(right_converted) - 1]

            right = right_converted

            if '-' in left and int(left) == 0:
                return u'%s %s %s %s %s' % (
                    self.negword,
                    self._int2word(np.int(left)),
                    self.pointword,
                    FRACTION[len(right)],
                    self._int2word(np.int(right))
                )
            else:
                return u'%s %s %s %s' % (
                    self._int2word(np.int(left)),
                    self.pointword,
                    FRACTION[len(right)],
                    self._int2word(np.int(right))
                )
        else:
            return self._int2word(np.int(n))

    def pluralize(self, n, form):
        return form

    def _cents_verbose(self, number, currency):
        return self._int2word(number, currency == 'KZT')

    def _int2word(self, n, feminine=False):
        if n < 0:
            return ' '.join([self.negword, self._int2word(abs(n))])

        if n == 0:
            return ZERO

        words = []
        chunks = list(splitbyx(str(n), 3))
        i = len(chunks)
        for x in chunks:
            i -= 1

            if x == 0:
                continue

            n1, n2, n3 = get_digits(x)

            if n3 > 0:
                if n3 > 1:
                    words.append(ONES[n3])
                words.append(HUNDRED)

            if n2 == 1:
                words.append(TEN)
            elif n2 > 1:
                words.append(TWENTIES[n2])

            if n1 > 0:
                words.append(ONES[n1])

            if i > 0:
                words.append(THOUSANDS[i])

        return ' '.join(words)

    def verify_ordinal(self, number):
        if not number == int(number):
            raise TypeError(self.errmsg_floatord % number)
        if not abs(number) == number:
            raise TypeError(self.errmsg_negord % number)

    def to_ordinal(self, number):
        self.verify_ordinal(number)
        word = self.to_cardinal(number)
        word_list = word.split(' ')
        word_list[-1] = ORDINALS[word_list[-1]]
        return " ".join(word_list)

    def to_time(self, number):
        word_list = []
        if len(number) == 2:
            word_list.append(self.to_cardinal(number[0]))
            word_list.append(self.to_cardinal(number[1]))
            return ' '.join(word_list)
        else:
            #print(number)
            word_list.append(self.to_cardinal(number[0]))
            word_list.append(self.to_cardinal(number[1]))
            word_list.append(self.to_cardinal(number[2]))
            return ' '.join(word_list)