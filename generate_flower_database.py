"""
Generates the comprehensive 102 flower database for PetalPop AI.
Maps Oxford 102 Flowers dataset categories (IDs 1-102 and 0-indexed 0-101)
to detailed, accurate botanical and care instructions.
"""

import json
import os

FLOWER_RAW_DATA = [
    (1, "Pink Primrose", "Oenothera speciosa",
     "A charming perennial with delicate saucer-shaped pale pink petals that open in the evening and soften in daylight.",
     "Moderate; water when top 1-2 inches of soil feel dry",
     "Full sun to partial shade (at least 6 hours daily)",
     "Well-draining, sandy or loamy soil with neutral pH",
     "Deadhead spent blossoms to encourage continuous summer blooming and prevent overly aggressive self-seeding."),

    (2, "Hard-leaved Pocket Orchid", "Paphiopedilum micranthum",
     "An exquisite lady's slipper orchid distinguished by its inflated, pouch-like pink labellum and tessellated foliage.",
     "Keep consistently moist with lukewarm distilled water; never allow potting medium to dry out completely",
     "Bright, filtered indirect sunlight; avoid direct scorching rays",
     "Specialized orchid mix consisting of fir bark, perlite, and sphagnum moss",
     "Maintain high humidity (60-80%) and good air circulation around the roots; fertilize weakly weekly."),

    (3, "Canterbury Bells", "Campanula medium",
     "A classic biennial cottage garden favorite boasting stately spires of bell-shaped violet, blue, or pink blossoms.",
     "Regular, deep watering; keep soil uniformly moist but not waterlogged",
     "Full sun in cooler climates; afternoon shade in warmer regions",
     "Moist, fertile, organically rich, well-draining garden soil",
     "Stake tall blooming spikes to protect against wind; cut stems down after blooming to encourage basal growth."),

    (4, "Sweet Pea", "Lathyrus odoratus",
     "A delightfully fragrant annual climber with butterfly-like winged flowers in delicate pastel shades.",
     "Generous and consistent watering, especially during hot summer blooming periods",
     "Full sun with roots shaded or mulched to keep them cool",
     "Rich, deeply cultivated, humus-rich soil with good drainage",
     "Pinch young seedlings when 4 inches tall to stimulate branching; pick flowers frequently to prolong the season."),

    (5, "English Marigold", "Calendula officinalis",
     "A cheerful, hardy annual featuring vibrant golden-orange daisy blossoms revered for centuries in herbal medicine.",
     "Low to moderate; drought tolerant once established, avoid overwatering",
     "Full sun for maximum blossom density and color intensity",
     "Average, well-drained garden soil; tolerant of poor soils",
     "Deadhead regularly to keep flowering continuous; leaves and petals are edible and attract beneficial pollinators."),

    (6, "Tiger Lily", "Lilium lancifolium",
     "A dramatic Asiatic perennial displaying recurved orange petals dotted with prominent dark purple speckles.",
     "Deep weekly watering during active growth; allow slight surface drying between waterings",
     "Full sun to light afternoon shade; prefers cool roots",
     "Moist, acidic to neutral, rich, porous and well-draining soil",
     "Mulch around the base to preserve moisture; protect emerging spring shoots from slugs and lily beetles."),

    (7, "Moon Orchid", "Phalaenopsis amabilis",
     "A graceful moth orchid with long arching sprays of pure white, sculpted blossoms; Indonesia's national flower.",
     "Water thoroughly every 7-10 days when root tips turn silvery; allow medium to dry slightly",
     "Medium to bright indirect light (east or shaded south window)",
     "Coarse bark mix with perlite or charcoal; excellent aeration",
     "Never leave standing water in the leaf crown; wipe leaves gently with a damp cloth to remove dust."),

    (8, "Bird of Paradise", "Strelitzia reginae",
     "An iconic tropical perennial with spectacular crane-like floral bracts in vivid orange, royal blue, and white.",
     "Moderate; keep moist during spring and summer, reduce watering during winter rest",
     "Bright direct sunlight (at least 4-6 hours daily for reliable blooming)",
     "Rich, loamy, well-aerated potting mix with excellent drainage",
     "Feed monthly during active spring-summer growth with a balanced liquid fertilizer; protect from frost."),

    (9, "Monkshood", "Aconitum napellus",
     "A statuesque woodland perennial crowned with hooded, deep violet-blue florets resembling monastic cowls.",
     "Consistent, even moisture; does not tolerate dry periods",
     "Partial shade to dappled sunlight; protect from harsh midday heat",
     "Cool, moisture-retentive, humus-rich soil",
     "All parts of the plant are toxic; wear gloves when handling and keep away from pets and small children."),

    (10, "Globe Thistle", "Echinops ritro",
     "A striking architectural perennial featuring spherical metallic-blue floral globes atop silver-green foliage.",
     "Very low water requirements; highly drought tolerant once rooted",
     "Full direct sunlight for sturdy stems and vivid blue coloration",
     "Poor to moderately fertile, dry to medium, sharp-draining soil",
     "Cut back faded flower heads to encourage a secondary autumn flush; outstanding as dried flowers."),

    (11, "Snapdragon", "Antirrhinum majus",
     "A classic upright garden favorite with whimsical, two-lipped blossoms that open like dragon jaws when gently squeezed.",
     "Water regularly until established; allow top inch of soil to dry between waterings",
     "Full sun; tolerates partial afternoon shade in warm zones",
     "Nutrient-rich, well-drained soil amended with compost",
     "Pinch early growth tips to create bushier branching plants; deadhead spent spires regularly."),

    (12, "Colt's Foot", "Tussilago farfara",
     "An early spring wild perennial whose golden dandelion-like flower heads emerge before its hoof-shaped leaves.",
     "Moderate to high; naturally thrives near stream banks and damp roadsides",
     "Full sun to partial woodland shade",
     "Heavy clay, moist loams, or alkaline calcareous soils",
     "Historically used in traditional European herbal teas; control spreading rhizomes in cultivated beds."),

    (13, "King Protea", "Protea cynaroides",
     "The majestic national flower of South Africa, famous for its colossal artichoke-like floral bowl and pink bracts.",
     "Low to moderate; deeply sensitive to waterlogging; deep infrequent soakings",
     "Full sun with strong, fresh air circulation",
     "Acidic, nutrient-poor, rocky or sandy soil with zero phosphorus fertilizer",
     "Never cultivate around surface roots; water early morning and avoid wetting foliage to prevent fungal blight."),

    (14, "Spear Thistle", "Cirsium vulgare",
     "The proud national emblem of Scotland, presenting bristling spiny bracts capped by rich purple-magenta brushes.",
     "Minimal watering; highly drought resistant",
     "Full sun",
     "Adaptable to virtually any well-drained soil, including gravel and poor clay",
     "Frequented by bumblebees, butterflies, and goldfinches; prune seed heads if you wish to prevent spreading."),

    (15, "Yellow Iris", "Iris pseudacorus",
     "A vigorous wetland perennial boasting bright sunshine-yellow fleur-de-lis flowers along waterside margins.",
     "High; thrives in shallow standing water or permanently saturated boggy ground",
     "Full sun to dappled wetland shade",
     "Rich, boggy, heavy clay or waterlogged soils",
     "Ideal for pond borders, rain gardens, and water purification zones; divide rhizomes every 3-4 years."),

    (16, "Globe-Flower", "Trollius europaeus",
     "A cool-climate buttercup relative with luminous, spherical butter-yellow blossoms glowing against dark dissected leaves.",
     "High moisture; keep soil constantly damp and cool",
     "Full sun in cool climates; partial shade in warm regions",
     "Deep, moist, humus-rich soil; thrives near streams and damp meadows",
     "Mulch generously to conserve soil moisture; cut stems back after blooming for fresh lush foliage."),

    (17, "Purple Coneflower", "Echinacea purpurea",
     "A resilient North American prairie perennial showcasing reflexed magenta-pink petals around a prominent spiky copper cone.",
     "Low; very drought-tolerant once established",
     "Full sun (6+ hours per day for strong, upright stems)",
     "Well-draining sandy loam; tolerates clay and lean soils",
     "Leave winter seed heads standing to feed goldfinches and provide seasonal garden architecture."),

    (18, "Peruvian Lily", "Alstroemeria aurea",
     "A long-lasting cut flower presenting clusters of azalea-like speckled flowers in warm sunset shades.",
     "Moderate; water deeply during blooming season, keeping soil slightly moist",
     "Full sun in morning, light shade in hot afternoons",
     "Rich, sandy-loam, slightly acidic, free-draining soil",
     "Pull rather than cut flowering stems from the base to stimulate dormant rhizome buds."),

    (19, "Balloon Flower", "Platycodon grandiflorus",
     "A delightful bellflower relative whose puffy, balloon-shaped buds pop open into starry five-pointed violet-blue blooms.",
     "Moderate; water regularly during first season; fairly drought-tolerant once established",
     "Full sun to partial light shade",
     "Loamy, well-draining, organically rich soil",
     "Mark planting spot in autumn, as shoots emerge late in spring; deadhead individual spent stars."),

    (20, "Giant White Arum Lily", "Zantedeschia aethiopica",
     "A sculptural evergreen perennial celebrated for its elegant, pure white funnel-shaped spathes and golden spadix.",
     "Ample water during spring and summer; can grow as a semi-aquatic marginal plant",
     "Full sun to semi-shade; protect from harsh afternoon sun in hot climates",
     "Rich, fertile, moisture-retentive, loamy or boggy soil",
     "Protect crown from heavy winter frosts with straw mulch; toxic if ingested by pets."),

    (21, "Fire Lily", "Gloriosa superba",
     "A breathtaking tropical climbing vine with swept-back, flame-like petals featuring wavy yellow-to-crimson margins.",
     "Regular moisture during active growth; reduce sharply when vine enters winter dormancy",
     "Bright indirect light or gentle morning sun",
     "Light, sandy, well-draining peat-based potting soil",
     "Handle tuberous roots gently; provide a supportive trellis for climbing leaf-tip tendrils."),

    (22, "Pincushion Flower", "Scabiosa columbaria",
     "A whimsical cottage perennial boasting soft lavender-blue domed heads bristling with pin-like white stamens.",
     "Moderate; water when top 2 inches dry out; avoid waterlogged roots",
     "Full sun",
     "Well-drained, neutral to slightly alkaline soil",
     "Deadhead frequently to encourage continuous flowering from early summer through first frost."),

    (23, "Fritillary", "Fritillaria meleagris",
     "Also known as the Snake's Head Fritillary, renowned for its hypnotic chequered bell blossoms in shades of purple and slate.",
     "Moist conditions; do not let the soil bake dry during summer dormancy",
     "Partial shade or dappled light under deciduous trees",
     "Humus-rich, moisture-retentive, well-draining meadow soil",
     "Plant fragile bulbs on their sides to prevent water from collecting in the hollow crown."),

    (24, "Red Ginger", "Alpinia purpurata",
     "A lush tropical perennial bearing vivid crimson-red, long-lasting cone bracts with tiny white true flowers tucked inside.",
     "High moisture; water frequently and mist to maintain elevated tropical humidity",
     "Bright indirect sunlight or partial dappled shade",
     "Rich, organic, highly moisture-retentive, well-draining soil",
     "Apply a balanced tropical fertilizer every 6 weeks during spring and summer; protect from chilly drafts."),

    (25, "Grape Hyacinth", "Muscari armeniacum",
     "A sweet, early-spring bulb displaying tight, conical clusters of cobalt-blue bead-like flowers resembling mini grapes.",
     "Moderate water during active spring growth; prefers drier soil during summer dormancy",
     "Full sun to partial spring shade beneath deciduous canopies",
     "Average, well-draining garden soil",
     "Plant bulbs in autumn in dense drifts; naturalizes effortlessly and returns year after year."),

    (26, "Corn Poppy", "Papaver rhoeas",
     "The iconic Flanders poppy with silky, paper-thin scarlet petals accented by dark central stamens.",
     "Low to moderate; water lightly; handles drought well once established",
     "Full sun",
     "Poor to moderately fertile, well-drained, sandy or gravelly soil",
     "Sow seeds directly where they are to grow as taproots resent transplanting; deadhead for prolonged blooms."),

    (27, "Prince of Wales Feathers", "Celosia argentea",
     "A vibrant annual featuring dramatic, velvety feather-like plumes in incandescent shades of crimson, pink, and gold.",
     "Moderate watering; keep soil moist but never soggy to prevent stem rot",
     "Full sun (6-8 hours daily)",
     "Fertile, rich, well-draining soil with plenty of organic matter",
     "Stunning in fresh arrangements and dries with vivid color retention; fertilize every 3 weeks."),

    (28, "Stemless Gentian", "Gentiana acaulis",
     "An alpine jewel forming evergreen mats of glossy green rosettes topped with large, trumpet-shaped true azure flowers.",
     "Consistently moist; requires good moisture during summer but excellent winter drainage",
     "Full sun in cool mountain areas; light partial shade in warm zones",
     "Gritty, humus-rich, acidic to neutral, alpine scree soil",
     "Protect from scorching summer heat; ideal for rock gardens, trough planters, and stone walls."),

    (29, "Artichoke", "Cynara cardunculus",
     "A monumental architectural plant with silvery thistle foliage and massive edible buds that burst into electric violet flower heads.",
     "Regular deep watering, especially during warm flowering weather",
     "Full sun",
     "Deep, rich, fertile, well-drained loam",
     "Mulch heavily in cold winter zones; harvest flower buds young for eating, or allow them to open for spectacular blooms."),

    (30, "Sweet William", "Dianthus barbatus",
     "A beloved heirloom biennial with dense, fragrant flattop clusters of fringed blossoms in striking bicolors.",
     "Moderate; water at the base to keep foliage dry and prevent fungal leaf spot",
     "Full sun to partial morning sun",
     "Slightly alkaline, fertile, well-draining garden soil",
     "Deadhead spent clusters to prolong blooming period; readily self-sows in garden borders."),

    (31, "Carnation", "Dianthus caryophyllus",
     "A timeless classic flower with ruffled, clove-scented petals in a rainbow of pastel pinks, whites, and reds.",
     "Allow top 1-2 inches of soil to dry between waterings; avoid overwatering",
     "Bright sun (minimum 5 hours direct sunshine daily)",
     "Well-draining, slightly alkaline soil enriched with compost",
     "Pinch off lateral buds if you wish to cultivate one large terminal show blossom per stem."),

    (32, "Garden Phlox", "Phlox paniculata",
     "A summer border staple producing fragrant pyramidal clusters of five-lobed blossoms that attract butterflies.",
     "Regular deep watering at the root zone (avoid wetting leaves)",
     "Full sun to light partial afternoon shade",
     "Moist, humus-rich, fertile, well-draining soil",
     "Provide good air circulation and water at soil level to minimize powdery mildew susceptibility."),

    (33, "Love in the Mist", "Nigella damascena",
     "An ethereal cottage garden annual with delicate cornflower-blue petals nestled within a feathery web of fine filigree foliage.",
     "Low to moderate; drought tolerant once established",
     "Full sun",
     "Average, well-draining garden soil; tolerates poor soils",
     "Followed by intriguing balloon-shaped horned seed capsules prized in dried floral arrangements."),

    (34, "Mexican Aster (Cosmos)", "Cosmos bipinnatus",
     "An effortlessly cheerful daisy-like annual with delicate feathery leaves and silky pink, white, or carmine ray petals.",
     "Low; water only during prolonged dry spells",
     "Full sun for prolific flowering",
     "Poor to moderately fertile, well-drained soil (overly rich soil causes excessive foliage and fewer flowers)",
     "Deadhead regularly; pinch young plants when 12 inches tall for compact, bushy floral architecture."),

    (35, "Alpine Sea Holly", "Eryngium alpinum",
     "A prized alpine specimen presenting steel-blue thimble flower heads surrounded by intricate lace-like metallic bracts.",
     "Low; highly drought-tolerant with deep taproot system",
     "Full sun",
     "Dry, sandy, gravelly, sharply-drained soil",
     "Avoid disturbing mature plants because of their deep taproot; excellent dried flower candidate."),

    (36, "Ruby-lipped Cattleya", "Cattleya labiata",
     "The queen of tropical orchids, boasting large, sumptuous lilac petals contrasted with a velvety crimson ruffled lip.",
     "Water thoroughly, then allow bark medium to dry out almost completely before watering again",
     "Bright, filtered light (strong indirect sunlight with slight morning sun)",
     "Coarse fir bark potting mix with perlite and horticultural charcoal",
     "Provide day/night temperature differential (10-15 deg F) to stimulate flower sheath induction."),

    (37, "Cape Flower", "Pelargonium peltatum",
     "A trailing evergreen beauty producing clusters of delicate pink to crimson flowers and fleshy ivy-shaped leaves.",
     "Moderate; allow top half of potting mix to dry between waterings",
     "Full sun to light afternoon shade",
     "Well-draining, lightweight, loamy potting compost",
     "Ideal for hanging baskets and window boxes; feed every two weeks with high-potassium plant food."),

    (38, "Great Masterwort", "Astrantia major",
     "A romantic woodland perennial with intricate star-like umbels surrounded by papery, collar-like floral bracts.",
     "High moisture; needs moist soil and suffers if dried out",
     "Partial shade; will tolerate full sun if soil remains continuously moist",
     "Rich, organic, moisture-retentive, fertile soil",
     "Cut back hard after initial blooming flush to stimulate a vigorous second crop of late summer flowers."),

    (39, "Siam Tulip", "Curcuma alismatifolia",
     "A tropical ginger family marvel sporting elegant tulip-like pink or lilac bracts atop upright lanceolate foliage.",
     "Consistent moisture while growing; sharply reduce in winter dormancy",
     "Bright filtered sunlight to light morning sun",
     "Rich, porous, peat-based potting soil with excellent drainage",
     "In cold climates, lift dormant rhizomes in autumn and store in dry peat moss at 55 deg F."),

    (40, "Lenten Rose", "Helleborus orientalis",
     "An aristocrat of late winter and early spring, presenting nodding cup-shaped blooms in dusty pink, slate, and plum.",
     "Moderate; prefers evenly moist soil, tolerant of summer dry periods once established",
     "Partial to full woodland shade; sheltered from drying winds",
     "Rich, fertile, humus-heavy, neutral to alkaline, well-drained soil",
     "Trim away old winter foliage just as floral buds emerge from the soil to showcase the blossoms cleanly."),

    (41, "Barbeton Daisy (Gerbera)", "Gerbera jamesonii",
     "A radiant South African daisy displaying bold, saturated ray petals radiating from a luminous central disc.",
     "Water thoroughly when top inch is dry; never saturate the plant's crown",
     "Bright direct morning sun; light filtered afternoon light",
     "Rich, sandy loam with fast drainage; slightly acidic pH",
     "Remove spent flower stems down at the crown to stimulate continuous new buds; fertilize every 2 weeks."),

    (42, "Daffodil", "Narcissus pseudonarcissus",
     "The herald of springtime, bearing sunny yellow trumpet coronas encircled by six spreading petals.",
     "Moderate water during spring growth; keep soil dry during summer dormancy",
     "Full sun to light deciduous tree shade",
     "Well-draining, fertile garden soil",
     "Allow foliage to turn completely yellow and wither naturally before removing to nourish next year's bulb."),

    (43, "Sword Lily (Gladiolus)", "Gladiolus hortulanus",
     "A dramatic summer bloomer featuring tall, sword-shaped leaves and dense spikes of trumpet-like ruffled florets.",
     "Regular deep watering (about 1 inch per week during bud development)",
     "Full sun",
     "Light, sandy-loam, well-draining soil",
     "Stake tall cultivars early; in frost-prone zones, dig corms in autumn and store cool and dry."),

    (44, "Poinsettia", "Euphorbia pulcherrima",
     "A holiday classic showcasing dramatic velvety scarlet floral bracts surrounding subtle central yellow cyathia.",
     "Water when surface soil feels dry to the touch; drain excess water completely",
     "Bright indirect sunlight; shield from direct scorching sun and drafty doors",
     "Loose, porous potting mix with peat and perlite",
     "To encourage winter re-blooming, provide 14 hours of total uninterrupted darkness nightly for 8-10 weeks in autumn."),

    (45, "Bolero Deep Blue (Lisianthus)", "Eustoma grandiflorum",
     "An elegant rose-like cut flower with spiraled velvety royal purple-blue petals and soft gray-green leaves.",
     "Keep consistently moist without saturating; sensitive to both waterlogging and drought",
     "Full sun to partial morning sun",
     "Rich, well-draining loam with a neutral to slightly alkaline pH",
     "Water strictly at base level to protect delicate petals from spotting; support slender flowering stems."),

    (46, "Wallflower", "Erysimum cheiri",
     "A wonderfully scented spring biennial bearing rich golden, bronze, or orange blossoms along old stone garden walls.",
     "Low to moderate; remarkably drought tolerant once established",
     "Full sun",
     "Dry, well-drained, alkaline or gravelly soil",
     "Pinch back young growing tips to promote bushy growth; thrives in rock crevices and mortar joints."),

    (47, "Marigold", "Tagetes erecta",
     "An energetic, pungent annual with dense carnation-like pompons of vivid tangerine and sunny yellow.",
     "Moderate; water at the base once weekly or when soil feels dry",
     "Full direct sunlight",
     "Average, well-draining garden soil; tolerant of poor ground",
     "Natural pest deterrent; companion plant next to vegetable gardens to repel nematodes and pests."),

    (48, "Buttercup", "Ranunculus acris",
     "A gleaming meadow wildflower with glossy, varnished yellow petals that reflect light like polished lacquer.",
     "Moderate to high; prefers damp, moisture-retentive ground",
     "Full sun to light partial shade",
     "Moist, heavy loam or clay soil",
     "Naturally spreads through meadows; beautiful in naturalized lawn areas and wildflower sanctuaries."),

    (49, "Oxeye Daisy", "Leucanthemum vulgare",
     "A classic, innocent meadow wildflower with sparkling clean white ray petals surrounding a sunny yellow disk.",
     "Low to moderate; handles drought well once established",
     "Full sun to light shade",
     "Average, well-drained garden or meadow soil",
     "Deadhead spent blooms to encourage ongoing flowering; divide clumps every 2 to 3 years."),

    (50, "Common Dandelion", "Taraxacum officinale",
     "A resilient, nutrient-dense sunny yellow composite flower that matures into a delicate spherical seed clock.",
     "Minimal; deep taproots access underground moisture reserves",
     "Full sun to partial shade",
     "Adaptable to virtually any soil",
     "Every part is edible and medicinal; young spring leaves make bitter salads, roots make caffeine-free tea."),

    (51, "Petunia", "Petunia x hybrida",
     "A prolific bedding plant featuring trumpet-shaped blooms with velvety texture in an infinite array of colors.",
     "Regular deep watering; allow top inch to dry between waterings",
     "Full direct sun (minimum 6 hours daily)",
     "Light, fertile, well-draining potting soil",
     "Pinch back leggy stems mid-summer and apply high-phosphorus liquid fertilizer every 2 weeks for continuous blooms."),

    (52, "Wild Pansy", "Viola tricolor",
     "Also known as Heartsease, featuring charming tricolored smiling faces in deep purple, white, and yellow.",
     "Consistent moisture; keep soil evenly moist but never sodden",
     "Full sun in cool spring; partial shade during warmer summer days",
     "Rich, cool, humus-rich, moist, well-draining soil",
     "Edible blossoms make stunning salad and dessert garnishes; deadhead to promote continuous cool-season blooming."),

    (53, "Primula", "Primula obconica",
     "A sweet indoor and garden perennial producing dense umbels of pastel pink, lavender, and white blossoms.",
     "Keep compost uniformly moist; never allow root ball to dry out completely",
     "Cool, bright indirect light; avoid hot sunny windows",
     "Moist, organically rich, peat-based potting soil",
     "Deadhead faded flower stems down at the crown; keep at cooler ambient temperatures (55-65 deg F)."),

    (54, "Sunflower", "Helianthus annuus",
     "The ultimate symbol of sunshine, boasting towering thick stems crowned with colossal golden ray flower heads.",
     "Heavy water requirement during budding and flowering; water deeply at the base",
     "Full direct sun (needs 6-8+ hours of uninterrupted sunlight daily)",
     "Deep, rich, loose, nutrient-dense, well-draining soil",
     "Stake tall mammoth varieties in windy sites; harvest dry seed heads for roasting or feeding wild birds."),

    (55, "Pelargonium", "Pelargonium zonale",
     "A beloved zonal geranium featuring rounded scalloped leaves with horseshoe bands and ball-shaped floral clusters.",
     "Allow top 2 inches of soil to dry out between waterings; drought tolerant",
     "Full direct sunlight (at least 6 hours daily)",
     "Light, sandy, well-draining potting mixture",
     "Snap off faded flower stems cleanly at their branch joint; overwinter indoors in cold climates."),

    (56, "Bishop of Llandaff", "Dahlia 'Bishop of Llandaff'",
     "A celebrated heritage dahlia with dramatic dark bronze-black foliage and glowing semi-double scarlet-red blooms.",
     "Regular, deep watering; keep consistently moist during bud formation",
     "Full sun",
     "Rich, fertile, humus-laden, well-drained soil",
     "Pinch growing tips when 12 inches tall; lift tuberous roots after first autumn frost in cold regions."),

    (57, "Gaura", "Gaura lindheimeri",
     "Also known as Whirling Butterflies, producing wand-like arching stems of fluttery four-petaled pink and white blossoms.",
     "Low; very drought tolerant once established",
     "Full sun",
     "Well-draining, sandy or gravelly loam; handles poor soil",
     "Do not over-fertilize or stems may become floppy; shear back by half mid-summer for fresh autumn blossoms."),

    (58, "Meadow Geranium", "Geranium pratense",
     "A true hardy cranesbill perennial featuring saucer-shaped violet-blue blooms veined with delicate crimson threads.",
     "Moderate; keep soil moderately moist during peak flowering",
     "Full sun to light partial shade",
     "Average, moist, well-draining garden soil",
     "Shear back foliage hard to ground level after first flowering flush to stimulate fresh green leaves and a second bloom."),

    (59, "Orange Dahlia", "Dahlia pinnata",
     "A magnificent decorative dahlia producing geometric, layered flower heads in vivid fiery orange hues.",
     "Moderate to heavy; water 2-3 times weekly during hot dry spells",
     "Full direct sunlight",
     "Moist, fertile, loose, well-drained soil rich in organic compost",
     "Provide sturdy stakes early in development; disbud side buds if you wish to grow massive prize exhibition blooms."),

    (60, "Pink-Yellow Dahlia", "Dahlia variabilis",
     "A captivating bicolor dahlia exhibiting sunny lemon centers blending outwards into soft blush and candy pink petals.",
     "Regular, thorough watering; maintain consistently moist soil without pooling",
     "Full sun (6+ hours daily)",
     "Fertile, rich, well-draining loam",
     "Feed every two weeks with a low-nitrogen, high-potash liquid feed; deadhead spent blooms regularly."),

    (61, "Cautleya Spicata", "Cautleya spicata",
     "A striking Himalayan ginger with glossy reed-like green foliage, crimson leaf sheaths, and hooded bright yellow flowers.",
     "High moisture; needs abundant water throughout the summer growing period",
     "Partial shade or dappled woodland light; shield from direct midday sun",
     "Humus-rich, moist, fertile, moisture-retentive woodland soil",
     "Mulch deeply with autumn leaves in winter to protect tubers from frost; thrives beside shady pond borders."),

    (62, "Japanese Anemone", "Anemone hupehensis",
     "An elegant autumn-blooming perennial with tall, slender wiry stems carrying pure saucer-shaped pink or white blossoms.",
     "Moderate to high; keep soil consistently moist, especially during hot summer days",
     "Partial shade to morning sun",
     "Humus-rich, moist, well-draining, fertile soil",
     "Mulch well during the first winter after planting; clumps expand gradually to create dreamy late-season drifts."),

    (63, "Black-Eyed Susan", "Rudbeckia hirta",
     "A cheerful, hardy North American native displaying golden-orange rays circling a dark chocolate-brown cone.",
     "Low to moderate; tolerant of drought and heat once established",
     "Full sun",
     "Average, well-draining garden soil; adaptable to various soil types",
     "Deadhead to extend flowering from mid-summer into late autumn; seed cones feed finches throughout winter."),

    (64, "Silverbush", "Convolvulus cneorum",
     "A compact evergreen Mediterranean shrub covered in silky silver-sheened foliage and white trumpet blooms with golden eyes.",
     "Low; water sparingly; extremely sensitive to overwatering and root wetness",
     "Full direct sunlight",
     "Gritty, poor, sandy or rocky, sharply drained soil",
     "Trim lightly after flowering to maintain dense, rounded cushion habit; perfect for coastal and gravel gardens."),

    (65, "Californian Poppy", "Eschscholzia californica",
     "California's state flower, displaying satiny, cup-shaped golden-orange petals that close during overcast weather and nightfall.",
     "Low; very drought-tolerant once established",
     "Full sun",
     "Poor, sandy, well-drained soil; thrives in lean gravelly dirt",
     "Sow seeds directly in autumn or early spring; self-seeds reliably without ever becoming invasive."),

    (66, "Osteospermum (African Daisy)", "Osteospermum ecklonis",
     "A show-stopping daisy relative with radiant petals surrounding a metallic deep blue or purple central eye.",
     "Moderate; allow top 1-2 inches of soil to dry out between thorough waterings",
     "Full sun (needs direct light for flower heads to stay wide open)",
     "Light, sandy, well-draining soil",
     "Deadhead frequently to prolong flowering season; pinch young shoots for a neat, compact flowering mound."),

    (67, "Spring Crocus", "Crocus vernus",
     "A dainty early-spring herald poking through the snow with cup-like lilac, purple, and striped petals and glowing orange stigmas.",
     "Moderate water during spring growth; keep relatively dry during summer dormancy",
     "Full sun to light deciduous shade",
     "Sandy, gritty, well-draining soil",
     "Plant corms in autumn at 3-4 inches depth; naturalizes splendidly in grassy lawns and border edges."),

    (68, "Bearded Iris", "Iris germanica",
     "An iconic perennial with stately sword-like fans and complex blooms featuring upright standards and falling bearded petals.",
     "Low to moderate; water deeply during dry spring spells; drought-tolerant in summer",
     "Full sun (at least 6 hours daily)",
     "Fertile, neutral to slightly alkaline, sharply drained soil",
     "Plant rhizomes with their top surface exposed to sunshine; never bury rhizomes under thick mulch."),

    (69, "Windflower", "Anemone coronaria",
     "A stunning Mediterranean poppy-like flower displaying brilliant blue, scarlet, or violet petals with dark powdery centers.",
     "Moderate; keep moist while actively growing; allow to dry out after summer dieback",
     "Full sun to light partial shade",
     "Light, sandy, humus-enriched, well-draining soil",
     "Soak hard tubers in lukewarm water for several hours before planting in autumn or early spring."),

    (70, "Tree Poppy", "Romneya coulteri",
     "Also known as the Matilija Poppy or 'Fried Egg Plant', bearing enormous crinkled crepe-paper white petals around a fluffy yellow yolk.",
     "Very low; extremely drought tolerant once established; avoid overwatering",
     "Full sun with plenty of heat",
     "Dry, sandy, rocky, poor, sharply draining soil",
     "Cut back nearly to ground level in late autumn; can spread vigorously via underground suckering runners."),

    (71, "Gazania", "Gazania rigens",
     "Also called the Treasure Flower, presenting dazzling sunburst petals marked with contrasting dark bands around the base.",
     "Low; highly drought-resistant; water sparingly",
     "Full direct sunlight (blossoms remain closed on overcast days)",
     "Light, sandy, well-draining soil; thrives in dry coastal banks",
     "Deadhead spent flowers; excellent trailing or spreading groundcover for sunny, exposed retaining walls."),

    (72, "Azalea", "Rhododendron",
     "A magnificent spring-flowering shrub smothered in vibrant clusters of funnel-shaped blossoms from pale blush to blazing magenta.",
     "Consistent, regular moisture; use rainwater where possible (dislikes hard, limey tap water)",
     "Dappled shade or morning sun with afternoon canopy protection",
     "Acidic (pH 4.5-5.5), organic, moist, peaty, well-draining soil",
     "Mulch with pine needles or ericaceous compost; never plant too deeply as roots are fine and shallow."),

    (73, "Water Lily", "Nymphaea alba",
     "An enchanting aquatic perennial with floating round leaves and fragrant, multi-petaled star-like blossoms on the water surface.",
     "Submerged aquatic plant; requires calm, still pond water (12-36 inches depth)",
     "Full sun (minimum 5-6 hours uninterrupted sunshine for daily opening)",
     "Heavy garden loam or dedicated aquatic compost in perforated planting baskets",
     "Keep away from splashing fountain currents; divide overcrowded root tubers in late spring."),

    (74, "Rose", "Rosa",
     "The timeless queen of flowers, celebrated across cultures for its layered velvet petals and intoxicating romantic fragrance.",
     "Deep weekly watering at the base (approximately 1-2 inches weekly); avoid wetting foliage",
     "Full sun (at least 6 hours direct sunshine)",
     "Rich, fertile, deep, moisture-retentive loam enriched with well-rotted manure",
     "Prune annually in late winter to stimulate vigorous new flowering wood; feed with specialized rose fertilizer."),

    (75, "Thorn Apple", "Datura stramonium",
     "A dramatic nightshade featuring fragrant, upright trumpet-shaped white to lavender blossoms and spiky walnut-sized seed capsules.",
     "Moderate; water when top inch of soil feels dry",
     "Full sun to light shade",
     "Rich, well-drained, loamy garden soil",
     "All parts contain potent tropane alkaloids and are highly toxic; handle with gloves and keep away from children/pets."),

    (76, "Morning Glory", "Ipomoea purpurea",
     "A vigorous annual twining vine with heart-shaped leaves and satiny trumpet blossoms that unfurl at dawn and fade by midday.",
     "Moderate; keep moist until established, then moderately drought-tolerant",
     "Full direct sun",
     "Average to poor, well-draining soil (excessive fertilizer causes lush leaves but fewer blooms)",
     "Nick seed coats and soak in warm water for 24 hours prior to sowing; provide a trellis or arbor for climbing."),

    (77, "Passion Flower", "Passiflora caerulea",
     "An otherworldly tropical climber featuring intricate blue-and-white coronas, five stamens, and three purple stigmas.",
     "Regular watering during warm blooming months; reduce watering in winter",
     "Full sun to light partial shade in warm microclimates",
     "Moist, fertile, well-draining soil with good organic content",
     "Train vigorous vines against sunny south-facing walls or sturdy trellises; prune in spring to control size."),

    (78, "Lotus", "Nelumbo nucifera",
     "A revered sacred aquatic flower with immaculate water-repellent leaves and luminous pink blossoms that rise above murky water.",
     "Submerged aquatic; requires 8-18 inches of still water over heavy root soil",
     "Full sun (needs high light and warm temperatures to blossom)",
     "Heavy, rich, clay loam aquatic substrate",
     "Protect tuberous rhizomes from freezing solid in winter; leaves exhibit the famous self-cleaning 'lotus effect'."),

    (79, "Toad Lily", "Tricyrtis hirta",
     "A shade-loving autumn perennial showcasing orchid-like white blossoms heavily speckled with dark amethyst spots.",
     "High moisture; needs consistently damp, cool soil",
     "Partial to full woodland shade; sheltered from drying winds",
     "Humus-rich, organically moist, acidic, well-draining soil",
     "Plant close to garden pathways where its intricate, miniature speckled petals can be appreciated up close."),

    (80, "Anthurium", "Anthurium andraeanum",
     "Also known as Flamingo Flower, famous for its long-lasting, glossy heart-shaped red or pink spathes and yellow spadix.",
     "Water thoroughly when top 50% of potting mix is dry; sensitive to overwatering and root rot",
     "Bright, indirect filtered light (no direct scorching sun)",
     "Coarse, chunky, highly aerated mix (equal parts orchid bark, peat, and perlite)",
     "Maintain high room humidity (above 60%); wipe glossy foliage clean of dust regularly."),

    (81, "Frangipani (Plumeria)", "Plumeria rubra",
     "A divine tropical tree with fragrant spiral-shaped petals in white, yellow, and blush, iconic in Hawaiian leis.",
     "Water deeply when soil dries out during spring and summer; stop watering entirely when dormant in winter",
     "Full direct sunlight (at least 6 hours daily)",
     "Coarse, cactus-like, fast-draining potting soil",
     "Protect strictly from frost; prune leggy branches in early spring before fresh leaf growth starts."),

    (82, "Clematis", "Clematis armandii",
     "The king of flowering climbers, presenting fragrant clusters of starry white or almond-scented blossoms on evergreen vines.",
     "Regular deep watering; follow the classic rule: 'head in the sun, roots in the cool shade'",
     "Full sun to partial shade for foliage; base shaded with mulch or companion plants",
     "Fertile, moist, well-drained, alkaline to neutral soil",
     "Prune lightly after spring flowering to tidy up vines and remove tangled stems."),

    (83, "Hibiscus", "Hibiscus rosa-sinensis",
     "A glamorous tropical shrub presenting dramatic, saucer-sized trumpet flowers with prominent protruding pollen columns.",
     "Generous watering during warm blooming periods; keep soil consistently moist but never soggy",
     "Full sun to high bright indirect light indoors",
     "Rich, loamy, slightly acidic potting compost with good drainage",
     "Feed every two weeks with high-potassium fertilizer during flowering season; protect from temperatures below 50 deg F."),

    (84, "Columbine", "Aquilegia vulgaris",
     "A charming woodland perennial with intricate nodding flowers sporting elongated backward-pointing nectar spurs.",
     "Moderate; keep moist in spring, fairly drought-tolerant after blooming",
     "Partial shade; will tolerate full sun in cooler climates",
     "Rich, moist, well-draining garden soil",
     "Foliage may decline in hot midsummer; shear back to ground to stimulate a fresh flush of basal leaves."),

    (85, "Desert Rose", "Adenium obesum",
     "A succulent bonsai-like shrub with a swollen sculptural caudex stem topped with clusters of brilliant scarlet and pink trumpets.",
     "Water thoroughly only when potting mix is completely dry; minimize watering in winter dormancy",
     "Full direct blazing sunlight",
     "Extremely porous cactus and succulent mix with pumice and coarse grit",
     "Avoid overpotting; never allow plant to sit in drainage water; protect from temperatures below 50 deg F."),

    (86, "Tree Mallow", "Malva arborea",
     "A robust coastal sub-shrub with velvety palmate leaves and dense clusters of rich purple saucer-shaped flowers.",
     "Low to moderate; highly tolerant of drought, salt spray, and windy maritime conditions",
     "Full sun",
     "Well-drained, sandy or gravelly soil",
     "Prune back hard in early spring to maintain a dense, bushy form; excellent windbreak plant for coastal gardens."),

    (87, "Magnolia", "Magnolia grandiflora",
     "A magnificent aristocratic tree bearing glossy leathery evergreen leaves and enormous lemon-scented creamy-white blossoms.",
     "Moderate; deep supplemental water during hot dry summers for young trees",
     "Full sun to partial light shade",
     "Rich, organically fertile, slightly acidic, moisture-retentive, well-drained soil",
     "Avoid root disturbance as fleshy magnolia roots grow close to the soil surface; mulch generously."),

    (88, "Cyclamen", "Cyclamen persicum",
     "A delicate winter-blooming tuber with swept-back butterfly-like petals hovering above heart-shaped silver-patterned leaves.",
     "Water from the bottom when compost feels slightly dry; avoid pouring water directly into the center of the tuber",
     "Cool, bright indirect light; thrives in temperatures between 50-65 deg F",
     "Free-draining peat-based potting soil",
     "Allow plant to go dormant in summer by withholding water; resume watering in autumn when fresh leaves sprout."),

    (89, "Watercress", "Nasturtium officinale",
     "A rapid-growing aquatic perennial producing crisp, peppery edible leaves and tiny starry four-petaled white flowers.",
     "Extremely high; requires running or continuously circulating fresh water",
     "Full sun to partial shade",
     "Wet, organically rich mud, gravel, or shallow cool running stream beds",
     "Harvest leafy tops regularly to promote bushy regrowth; prized culinary ingredient for salads, soups, and sandwiches."),

    (90, "Canna Lily", "Canna indica",
     "A bold tropical perennial boasting colossal banana-like foliage and vivid spikes of gladiolus-like red, orange, and yellow blooms.",
     "Generous, deep watering; prefers constantly moist soil throughout summer",
     "Full sun (needs heat and sunlight for maximal flowering)",
     "Rich, fertile, humus-heavy garden soil with copious compost",
     "In cold winter climates, dig up rhizomes after frost blackens foliage, dry them, and store cool indoors."),

    (91, "Hippeastrum (Amaryllis)", "Hippeastrum hybridum",
     "A sensational holiday bulb producing colossal trumpet-shaped blooms atop thick, hollow green flower scapes.",
     "Water sparingly until the flower scape appears, then increase to regular moderate watering",
     "Bright indirect light to gentle morning sunshine",
     "Dense, well-draining potting compost; pot bulb with top third sitting above soil line",
     "After blooms fade, cut flower stalk and continue feeding leaves to recharge bulb for next season's bloom cycle."),

    (92, "Bee Balm", "Monarda didyma",
     "An aromatic North American herb bearing shaggy, crown-like whorls of scarlet-red blossoms beloved by hummingbirds.",
     "Moderate to high; keep soil evenly moist; dislikes prolonged drought",
     "Full sun to light partial shade",
     "Moist, humus-rich, fertile garden soil",
     "Divide clumps every 3 years to maintain plant vigor; leaves emit a delicious bergamot-citrus scent."),

    (93, "Ball Moss", "Tillandsia recurvata",
     "A fascinating epiphyte forming rounded silver-gray clusters that perch harmlessly on tree limbs and telephone lines.",
     "Low; absorbs ambient atmospheric moisture and rain; mist every 1-2 weeks in dry indoor environments",
     "Bright filtered indirect light to light shade",
     "None required (true air plant); mount on cork bark, driftwood, or wire",
     "Not a parasite; provide excellent air circulation and dunk in rainwater occasionally for optimal health."),

    (94, "Foxglove", "Digitalis purpurea",
     "A majestic woodland biennial carrying tall, dramatic spires lined with speckled tubular thimble blossoms.",
     "Consistent, regular moisture; keep soil evenly damp",
     "Partial to dappled woodland shade; tolerates full sun in cooler regions",
     "Humus-rich, moist, fertile, slightly acidic, well-drained soil",
     "All parts are toxic and contain digitalis cardiac glycosides; wear gloves when handling; readily self-sows."),

    (95, "Bougainvillea", "Bougainvillea spectabilis",
     "A vigorous woody climbing vine known for its incandescent papery bracts in magenta, purple, and coral.",
     "Low to moderate; water deeply, then let soil dry out between waterings; thrives on slight stress",
     "Full direct blazing sun (at least 6 hours daily for vivid bract color)",
     "Well-draining, gritty, sandy loam; sensitive to soggy roots",
     "Prune back hard after flowering cycles to shape vine; thorny stems provide excellent natural privacy barriers."),

    (96, "Camellia", "Camellia japonica",
     "The rose of winter, displaying glossy evergreen foliage and formal, symmetrical rose-like blooms in shades of pink, white, and red.",
     "Consistent, even moisture; never allow root ball to dry out completely",
     "Sheltered partial shade; protection from harsh early-morning winter sun and bitter cold winds",
     "Acidic (pH 5.0-6.0), organic, humus-rich, well-draining soil",
     "Mulch with pine bark to conserve root coolness; avoid planting deeply as root system is shallow."),

    (97, "Mallow", "Malva sylvestris",
     "A charming wild cottage perennial bearing saucer-shaped mauve-purple petals accented by dark radiating veins.",
     "Low to moderate; drought tolerant once roots are established",
     "Full sun to light partial shade",
     "Average, well-drained garden soil; adaptable to poor ground",
     "Deadhead spent blossoms to prevent excessive self-sowing and encourage secondary autumn flowers."),

    (98, "Mexican Petunia", "Ruellia simplex",
     "A vigorous perennial sporting narrow willow-like dark green leaves and trumpet-shaped purple-violet flowers.",
     "High adaptability; thrives in both wet, boggy margins and dry conditions once established",
     "Full sun to partial shade; maximum blooms in full sun",
     "Adaptable to wet clay, standard garden loam, or sandy soils",
     "Can spread vigorously via seeds and underground rhizomes; grow in containers if you wish to restrict spreading."),

    (99, "Bromelia", "Bromelia pinguin",
     "A bold tropical terrestrial bromeliad with rosette of spiny sword leaves and a vibrant fiery-red flowering central heart.",
     "Low to moderate; fill central leaf rosette 'cup' with clean rainwater; water soil moderately",
     "Bright, sunny filtered light; handles strong sunshine",
     "Light, sandy, porous, well-draining soil or succulent blend",
     "Spiny leaf margins make it an impenetrable barrier; mother plant produces multiple offsets ('pups') after blooming."),

    (100, "Blanket Flower", "Gaillardia pulchella",
     "A cheerful, tough daisy featuring sunset bicolors of fiery crimson banded with golden yellow fringed tips.",
     "Low; extremely heat and drought tolerant",
     "Full sun",
     "Poor, sandy, well-draining, lean soil; avoid overly rich soil",
     "Deadhead regularly to maintain tidy flowering cushions; thrives in exposed, hot summer gardens."),

    (101, "Trumpet Creeper", "Campsis radicans",
     "A vigorous woody climber producing clusters of flaming orange-to-scarlet tubular flowers adored by hummingbirds.",
     "Low to moderate; remarkably drought tolerant once established",
     "Full sun for maximum flower production",
     "Adaptable to virtually any well-drained soil",
     "Extremely vigorous climber; provide sturdy fences or stone pergolas and prune aggressively each spring."),

    (102, "Blackberry Lily", "Iris domestica",
     "A unique iris relative with cheerful orange petals spotted with crimson, followed by seed pods resembling glossy ripe blackberries.",
     "Moderate; water regularly during active growth; drought tolerant once mature",
     "Full sun to light afternoon shade",
     "Rich, loamy, well-drained soil",
     "The exposed clusters of shiny black seeds persist well into winter and make striking elements in floral arrangements.")
]

def generate_flower_info_py():
    db = {}
    for item in FLOWER_RAW_DATA:
        cat_id, common, scientific, desc, water, sun, soil, care = item
        db[cat_id] = {
            "id": cat_id,
            "zero_index": cat_id - 1,
            "common_name": common,
            "scientific_name": scientific,
            "description": desc,
            "watering": water,
            "sunlight": sun,
            "soil": soil,
            "care_tips": care
        }
    
    code = f'''"""
PetalPop AI - Flower Botanical & Care Information Database
Comprehensive profiles for all 102 categories of the Oxford 102 Flowers dataset.
"""

from typing import Dict, Any, Optional

FLOWER_DATABASE: Dict[int, Dict[str, Any]] = {repr(db)}

# Index by 0-based index as well (0 to 101)
FLOWER_DATABASE_0_INDEXED: Dict[int, Dict[str, Any]] = {{
    info["zero_index"]: info for info in FLOWER_DATABASE.values()
}}

# Index by common name lowercase
FLOWER_BY_NAME: Dict[str, Dict[str, Any]] = {{
    info["common_name"].lower(): info for info in FLOWER_DATABASE.values()
}}

def get_flower_by_id(flower_id: int) -> Optional[Dict[str, Any]]:
    """Get flower details by Oxford 102 1-based class ID (1-102)."""
    return FLOWER_DATABASE.get(flower_id)

def get_flower_by_zero_index(zero_idx: int) -> Optional[Dict[str, Any]]:
    """Get flower details by model prediction 0-based index (0-101)."""
    return FLOWER_DATABASE_0_INDEXED.get(zero_idx)

def get_flower_by_name(name: str) -> Optional[Dict[str, Any]]:
    """Lookup flower details by common name."""
    return FLOWER_BY_NAME.get(name.lower().strip())

def get_all_flower_names() -> list[str]:
    """Return sorted list of all 102 flower common names."""
    return [FLOWER_DATABASE[i]["common_name"] for i in range(1, 103)]
'''
    os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"), exist_ok=True)
    target_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src", "flower_info.py")
    with open(target_path, "w", encoding="utf-8") as f:
        f.write(code)
    print(f"Generated {target_path} successfully with {len(db)} flower classes.")

if __name__ == "__main__":
    generate_flower_info_py()
