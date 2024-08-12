import cv2 as cv

def colorBackgroundText(img, text, font, scale, pos, thickness, color, background_color, pad_x=3, pad_y=3):
    (t_w, t_h), _ = cv.getTextSize(text, font, scale, thickness)
    x, y = pos
    cv.rectangle(img, (x - pad_x, y - pad_y), (x + t_w + pad_x, y + t_h + pad_y), background_color, -1)
    cv.putText(img, text, (x, y + t_h + pad_y), font, scale, color, thickness)

def textWithBackground(img, text, font, scale, pos, textColor=(255, 255, 255), bgColor=(0, 0, 0), bgOpacity=0.5, textThickness=2):
    (text_w, text_h), _ = cv.getTextSize(text, font, scale, textThickness)
    x, y = pos
    overlay = img.copy()
    cv.rectangle(overlay, (x, y), (x + text_w, y + text_h + int(scale * 10)), bgColor, -1)
    cv.addWeighted(overlay, bgOpacity, img, 1 - bgOpacity, 0, img)
    cv.putText(img, text, (x, y + text_h + int(scale * 5)), font, scale, textColor, textThickness)
    return img
