#[derive(Debug, Clone, Copy)]
struct OwnerSampleReplay {
    frame_size: Size<i32, Physical>,
    lane: usize,
    bounds: Rectangle<i32, Physical>,
    translation: Point<i32, Physical>,
    sample: [f32; 2],
    output_read: Rectangle<i32, Physical>,
}

impl OwnerSampleReplay {
    fn new(
        frame_size: Size<i32, Physical>,
        lane: usize,
        source: Rectangle<i32, Physical>,
        output_read: Rectangle<i32, Physical>,
    ) -> Option<Self> {
        let samples = [[0.375, 0.125], [0.875, 0.375], [0.125, 0.625], [0.625, 0.875]];
        let sample = *samples.get(lane)?;
        let origin = Point::from((
            (lane % 2) as i32 * source.size.w,
            (lane / 2) as i32 * source.size.h,
        ));
        Some(Self {
            frame_size,
            lane,
            bounds: Rectangle::new(origin, source.size),
            translation: origin - source.loc,
            sample,
            output_read,
        })
    }

    fn clip_output_region(self, region: Rectangle<i32, Physical>) -> Option<Rectangle<i32, Physical>> {
        region.intersection(self.output_read)
    }

    fn shift(self, resolved_texture: bool) -> [f32; 2] {
        [
            self.translation.x as f32
                + if resolved_texture {
                    0.0
                } else {
                    0.5 - self.sample[0]
                },
            self.translation.y as f32
                + if resolved_texture {
                    0.0
                } else {
                    0.5 - self.sample[1]
                },
        ]
    }

    fn damage(self, rects: Vec<Rectangle<i32, Physical>>) -> Vec<Rectangle<i32, Physical>> {
        rects
            .into_iter()
            .filter_map(|mut rect| {
                rect.loc += self.translation;
                rect.intersection(self.bounds)
            })
            .collect()
    }
}

#[cfg(test)]
mod owner_sample_tests {
    use super::*;
    #[test]
    fn each_lane_preserves_native_sample_and_exact_resolved_prefix_grid() {
        let source = Rectangle::new((17, 29).into(), (12, 10).into());
        for lane in 0..4 {
            let replay = OwnerSampleReplay::new((100, 80).into(), lane, source, source).unwrap();
            assert_eq!(replay.frame_size, (100, 80).into());
            assert_eq!(replay.lane, lane);
            assert_eq!(
                replay.clip_output_region(Rectangle::from_size((100, 80).into())),
                Some(source)
            );
            let origin = replay.bounds.loc;
            // A destination fragment centre minus the analytic viewport shift
            // is precisely its source pixel plus the canonical sample.
            let shift = replay.shift(false);
            assert_eq!(
                origin.x as f32 + 0.5 - shift[0],
                source.loc.x as f32 + replay.sample[0]
            );
            assert_eq!(
                origin.y as f32 + 0.5 - shift[1],
                source.loc.y as f32 + replay.sample[1]
            );
            let shift = replay.shift(true);
            assert_eq!(origin.x as f32 + 0.5 - shift[0], source.loc.x as f32 + 0.5);
        }
        assert!(OwnerSampleReplay::new((100, 80).into(), 4, source, source).is_none());
    }

    #[test]
    fn lane_damage_cannot_touch_an_adjacent_atlas_quadrant() {
        let replay = OwnerSampleReplay {
            frame_size: (100, 80).into(),
            lane: 3,
            bounds: Rectangle::new((12, 10).into(), (12, 10).into()),
            translation: (7, 3).into(),
            sample: [0.625, 0.875],
            output_read: Rectangle::from_size((100, 80).into()),
        };
        assert_eq!(
            replay.damage(vec![Rectangle::from_size((100, 80).into())]),
            vec![replay.bounds]
        );
        assert_eq!(replay.shift(false), [6.875, 2.625]);
        assert_eq!(replay.shift(true), [7.0, 3.0]);
    }
}
