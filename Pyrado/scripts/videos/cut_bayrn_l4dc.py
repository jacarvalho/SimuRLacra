"""
Script to cut videos of the experiments (on the Quanser Ball-Balancer and Quanser Cart-Pole) for the journal.
"""
import os.path as osp
import warnings
from moviepy.editor import *

# General setting
# moviesize = (1080, 1920)
font = 'LatinModernMath-Regular'  # Amiri-regular LatinModernMath-Regular
color = 'white'
opacity = 0.6
fontsize = 28
title_fontsize = 34
title_duration = 3  # seconds
pos_in_banner = (5, 'center')
# unexplained_offset = 3.0

# Create title slide
title_txt = """
Bayesian Domain Randomization (BayRn)
for Sim-to-Real Transfer


Fabio Muratore (1, 2),
Christian Eilers (1, 2),
Michael Gienger (2),
Jan Peters (1)

(1) Technical University Darmstadt, Germany
(2) Honda Research Institute Europe, Germany
"""

# Get the footage
video_dir = '/home/muratore/HESSENBOX-DA/Publications/L4DC_2019/videos'
bayrn_1 = VideoFileClip(osp.join(video_dir, 'hor_blanc.MOV'), audio=False)
bayrn_2 = VideoFileClip(osp.join(video_dir, 'hor_screw.MOV'), audio=False)
# The set_start() after CompositeVideoClip() is applied to all composed videos
bayrn_comp = CompositeVideoClip([bayrn_1.crossfadein(.3),
                               bayrn_2.set_start(bayrn_1.duration).crossfadein(.3)]). \
    set_duration(bayrn_1.duration + bayrn_2.duration
                 # + bayrn_3.duration
                 ).set_start(title_duration)

if not bayrn_1.size == bayrn_2.size:  # == bayrn_3.size:
    warnings.warn('Video sizes of the individual videos are not equal.')
w, h = bayrn_comp.size

# Clip with text and a semi-opaque black heading QBB
# pos argument in on_color() is the relative position in the banner
bayrn_txt = TextClip('Evaluating BayRn on the Furuta pendulum', font=font, color=color, fontsize=fontsize)
heading_bayrn = bayrn_txt.on_color(size=(w + bayrn_txt.w, bayrn_txt.h), color=(0, 0, 0), pos=pos_in_banner,
                               col_opacity=opacity). \
    set_duration(bayrn_comp.duration).set_start(title_duration). \
    set_pos((w*0.7, bayrn_txt.h)).crossfadein(.3)

bayrn_1_txt = TextClip('Nominal system', font=font, color=color, fontsize=fontsize, interline=6)
bayrn_1_txt = bayrn_1_txt.on_color(size=(w + bayrn_txt.w, bayrn_txt.h), color=(0, 0, 0), pos=pos_in_banner,
                               col_opacity=opacity). \
    set_duration(bayrn_1.duration). \
    set_start(title_duration). \
    set_pos((w*0.8, bayrn_txt.h + 2*bayrn_txt.h)).crossfadein(.3)

bayrn_2_txt = TextClip('Added a screw 2.7g screw', font=font, color=color, fontsize=fontsize, interline=6)
bayrn_2_txt = bayrn_2_txt.on_color(size=(w + bayrn_txt.w, bayrn_txt.h), color=(0, 0, 0), pos=pos_in_banner,
                               col_opacity=opacity). \
    set_duration(bayrn_2.duration). \
    set_start(title_duration + bayrn_1.duration). \
    set_pos((w*0.8, bayrn_txt.h + 2*bayrn_txt.h)).crossfadein(.3)

# Make the title when we know the size
title = TextClip(title_txt, size=bayrn_comp.size, font=font, fontsize=title_fontsize, color='white'). \
    set_pos('center').set_duration(title_duration).fadeout(.3)

# Final assembly
final = CompositeVideoClip(
    [title, bayrn_comp, heading_bayrn, bayrn_1_txt, bayrn_2_txt, ]).set_duration(title_duration + bayrn_comp.duration)
final.write_videofile(osp.join(video_dir, 'BayRn_evaluation_real_QQ.mov'), fps=30, codec='libx264', threads=8)
